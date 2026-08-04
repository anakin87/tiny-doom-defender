"""
Conv-stem "eye" in front of ModernBERT: the stem turns a frame stack into a fixed
1000-cell grid fed via `inputs_embeds`. Every cell is a real patch (no padding,
unlike a tokenized sequence), so the attention mask is all-ones.

Two heads sit on the shared trunk: DoomConvStemClassifier for SFT, ConvStemPolicy for RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from tiny_doom_defender.config import IN_CH, N_ACTION_STATES, N_PREV, RES_H, RES_W, STACK_PIXELS
from tiny_doom_defender.utils import check_stem_config

# =============================================================================
# The eye
# =============================================================================


class ConvStem(nn.Module):
    """3 RGB frames (9ch) @160x100 -> (B, 1000, hidden) sequence for ModernBERT.

    Two stride-2 3x3 convs downsample 160x100 -> 40x25 (the 1000-token grid). The
    last N_PREV actions are embedded and added to every cell so the model can
    subtract off ego-motion (the horizontal pan from turning): one embedding per
    inter-frame action, index-shifted per gap so the same action in different gaps
    stays distinct.
    """

    def __init__(self, in_ch=IN_CH, hidden=128, n_prev=N_PREV, n_action_states=N_ACTION_STATES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),  # 160x100 -> 80x50
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden, kernel_size=3, stride=2, padding=1),  # 80x50  -> 40x25
            nn.ReLU(inplace=True),
        )
        self.n_prev = n_prev
        self.n_action_states = n_action_states
        self.act_emb = nn.Embedding(n_prev * n_action_states, hidden)
        nn.init.normal_(self.act_emb.weight, std=0.02)

    def forward(self, frames, prev_actions):
        """frames: (B, 9, RES_H, RES_W) uint8 or float; prev_actions: (B, N_PREV) long."""
        if frames.dtype == torch.uint8:
            frames = frames.float()
        x = frames / 255.0
        x = self.conv(x)  # (B, hidden, 25, 40)
        x = x.flatten(2).transpose(1, 2)  # (B, 1000, hidden) row-major
        offsets = torch.arange(self.n_prev, device=prev_actions.device) * self.n_action_states
        act = self.act_emb(prev_actions.long() + offsets).sum(dim=1)  # (B, hidden)
        return x + act[:, None, :]


# =============================================================================
# Shared trunk: stem + encoder + attention pool
# =============================================================================


class _StemTrunk(nn.Module):
    """stem + ModernBERT encoder + learned attention pool -> pooled (B, hidden).

    We always feed `inputs_embeds`, so the encoder's token-embedding table is unused
    dead weight (create_model.py builds it with a tiny vocab to keep it small). The
    pool scores each of the 1000 cells with one learned linear, softmax-weights them,
    and sums.
    """

    def __init__(self, encoder_dir):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_dir)
        hidden = self.encoder.config.hidden_size
        # Stem geometry comes from config.py; the model dir's stem_config.json is the
        # stamp of what this encoder was built for, and only gets checked against it.
        check_stem_config(encoder_dir)
        self.stem = ConvStem(hidden=hidden)
        self.attn_weight = nn.Linear(hidden, 1, bias=False)
        self.hidden_size = hidden

    def encode(self, frames, prev_actions):
        embeds = self.stem(frames, prev_actions)  # (B, 1000, H)
        # All-ones: fixed grid, no padding — every cell is a real patch to attend to.
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        out = self.encoder(inputs_embeds=embeds, attention_mask=mask)
        token_embs = out.last_hidden_state  # (B, 1000, H)
        scores = self.attn_weight(token_embs).squeeze(-1)  # (B, 1000)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (token_embs * weights).sum(dim=1)  # (B, H)


# =============================================================================
# SFT classifier
# =============================================================================


class DoomConvStemClassifier(_StemTrunk):
    """Conv-stem ModernBERT with two independent heads for MultiDiscrete([3,2]).

    Turn (left/none/right) and shoot (no/yes) each get a linear head off the pooled
    vector; loss is the equal-weight sum of the two cross-entropies.
    """

    TURN_NAMES = ["turn_left", "turn_none", "turn_right"]
    SHOOT_NAMES = ["no_shoot", "shoot"]

    def __init__(self, encoder_dir):
        super().__init__(encoder_dir)
        self.turn_head = nn.Linear(self.hidden_size, 3)
        self.shoot_head = nn.Linear(self.hidden_size, 2)

    def forward(self, frames, prev_actions, turn_labels=None, shoot_labels=None):
        pooled = self.encode(frames, prev_actions)
        turn_logits = self.turn_head(pooled)
        shoot_logits = self.shoot_head(pooled)
        result = {"turn_logits": turn_logits, "shoot_logits": shoot_logits}
        if turn_labels is not None:
            loss_turn = F.cross_entropy(turn_logits, turn_labels)
            loss_shoot = F.cross_entropy(shoot_logits, shoot_labels)
            result["loss_turn"] = loss_turn
            result["loss_shoot"] = loss_shoot
            result["loss"] = loss_turn + loss_shoot
        return result


# =============================================================================
# RL policy (actor-critic)
# =============================================================================


def _sample_multidiscrete(logits_tuple, action=None):
    """Per-axis Categorical over the (B, n_i) logits -> (action (B, 2) long, log_prob (B,),
    entropy (B,)); log_prob and entropy summed over the axes."""
    dists = [torch.distributions.Categorical(logits=lg) for lg in logits_tuple]
    if action is None:
        actions = torch.stack([d.sample() for d in dists], dim=-1)
    else:
        actions = action.long()
    log_probs = torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(dists)], dim=-1).sum(dim=-1)
    entropies = torch.stack([d.entropy() for d in dists], dim=-1).sum(dim=-1)
    return actions, log_probs, entropies


class ConvStemPolicy(_StemTrunk):
    """Actor-critic over the conv-stem trunk, on the flat uint8 observation (B, OBS_LEN).

    unfreeze_blocks=0 trains everything; N>0 trains only the last N encoder layers +
    final_norm, with the stem frozen unless train_stem. The attention pool and the three
    heads are always trainable.
    """

    def __init__(self, encoder_dir, unfreeze_blocks=0, train_stem=False):
        super().__init__(encoder_dir)
        self.turn_head = nn.Linear(self.hidden_size, 3)
        self.shoot_head = nn.Linear(self.hidden_size, 2)
        self.value_head = nn.Linear(self.hidden_size, 1)
        for h in (self.turn_head, self.shoot_head, self.value_head):
            nn.init.orthogonal_(h.weight, gain=0.01)
            nn.init.zeros_(h.bias)

        n_layers = len(self.encoder.layers)
        if not 0 <= unfreeze_blocks <= n_layers:
            raise ValueError(f"unfreeze_blocks must be in [0, {n_layers}], got {unfreeze_blocks}")
        self._n_layers = n_layers
        self._stem_trainable = (unfreeze_blocks == 0) or train_stem
        if unfreeze_blocks == 0:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            self._encoder_trainable_idx = list(range(n_layers))
            self._embeddings_trainable = True
        else:
            split_idx = n_layers - unfreeze_blocks
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for i in range(split_idx, n_layers):
                for p in self.encoder.layers[i].parameters():
                    p.requires_grad_(True)
            for p in self.encoder.final_norm.parameters():
                p.requires_grad_(True)
            self._encoder_trainable_idx = list(range(split_idx, n_layers))
            self._embeddings_trainable = False
        for p in self.stem.parameters():
            p.requires_grad_(self._stem_trainable)

    def encoder_trainable_params(self):
        """Params for the encoder LR group: the encoder, plus the stem when trainable."""
        params = []
        if self._embeddings_trainable:
            params += [p for p in self.encoder.parameters() if p.requires_grad]
        else:
            for i in self._encoder_trainable_idx:
                params += list(self.encoder.layers[i].parameters())
            params += list(self.encoder.final_norm.parameters())
        if self._stem_trainable:
            params += list(self.stem.parameters())
        return params

    def head_trainable_params(self):
        """Params for the head LR group: attention pool + turn/shoot/value heads."""
        return (
            list(self.attn_weight.parameters())
            + list(self.turn_head.parameters())
            + list(self.shoot_head.parameters())
            + list(self.value_head.parameters())
        )

    @staticmethod
    def _unpack(obs):
        obs = obs.to(torch.uint8) if obs.dtype != torch.uint8 else obs
        frames = obs[:, :STACK_PIXELS].view(-1, IN_CH, RES_H, RES_W)
        prev = obs[:, STACK_PIXELS:].long()  # (B, N_PREV)
        return frames, prev

    def _heads(self, pooled):
        return (self.turn_head(pooled), self.shoot_head(pooled))

    def get_action_and_value(self, obs, action=None):
        frames, prev = self._unpack(obs)
        pooled = self.encode(frames, prev)
        logits = self._heads(pooled)
        value = self.value_head(pooled).squeeze(-1)
        action, log_prob, entropy = _sample_multidiscrete(logits, action=action)
        return action, log_prob, entropy, value

    def get_value(self, obs):
        frames, prev = self._unpack(obs)
        return self.value_head(self.encode(frames, prev)).squeeze(-1)

    def get_argmax_action(self, obs):
        frames, prev = self._unpack(obs)
        logits = self._heads(self.encode(frames, prev))
        return torch.stack([lg.argmax(-1) for lg in logits], dim=-1)

    def action_probs(self, obs):
        """(B, OBS_LEN) -> (turn_probs (B, 3), shoot_probs (B, 2)), for inspecting what the
        policy was considering rather than only what it chose."""
        frames, prev = self._unpack(obs)
        turn_logits, shoot_logits = self._heads(self.encode(frames, prev))
        return torch.softmax(turn_logits, -1), torch.softmax(shoot_logits, -1)
