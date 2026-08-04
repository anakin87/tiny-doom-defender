"""
Conv-stem "eye" in front of ModernBERT: the stem turns a frame stack into a fixed
1000-cell grid fed via `inputs_embeds`. Every cell is a real patch (no padding,
unlike a tokenized sequence), so the attention mask is all-ones.

Two heads sit on the shared trunk: DoomConvStemForActionClassification for SFT,
DoomConvStemPolicy for RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, ModernBertModel, PreTrainedModel

from tiny_doom_defender.configuration_doom import DoomConvStemConfig

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

    def __init__(self, config: DoomConvStemConfig):
        super().__init__()
        hidden = config.hidden_size
        self.conv = nn.Sequential(
            nn.Conv2d(config.in_channels, config.stem_channels, kernel_size=3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(config.stem_channels, hidden, kernel_size=3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
        )
        self.n_prev = config.n_prev_actions
        self.n_action_states = config.n_action_states
        self.act_emb = nn.Embedding(self.n_prev * self.n_action_states, hidden)

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


class DoomConvStemPreTrainedModel(PreTrainedModel):
    """stem + ModernBERT encoder + learned attention pool -> pooled (B, hidden).

    We always feed `inputs_embeds`, so the encoder's token-embedding table is unused
    dead weight (the config keeps its vocab tiny). The pool scores each of the 1000
    cells with one learned linear, softmax-weights them, and sums.
    """

    config_class = DoomConvStemConfig
    main_input_name = "frames"

    def __init__(self, config: DoomConvStemConfig):
        super().__init__(config)
        self.encoder = ModernBertModel(config.encoder_config)
        self.stem = ConvStem(config)
        self.attn_weight = nn.Linear(config.hidden_size, 1, bias=False)
        self.hidden_size = config.hidden_size
        # Flat-obs layout (the PPO vector env packs [frames | prev_actions] as uint8).
        self._stack_pixels = config.in_channels * config.res_h * config.res_w
        self._frame_shape = (config.in_channels, config.res_h, config.res_w)

    def _init_weights(self, module):
        # Only this model's own modules arrive here: the encoder is itself a
        # PreTrainedModel, so initialize_weights dispatches its modules to
        # ModernBERT's _init_weights.
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            module.reset_parameters()

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


class DoomConvStemForActionClassification(DoomConvStemPreTrainedModel):
    """Conv-stem ModernBERT with two independent heads for MultiDiscrete([3,2]).

    Turn (left/none/right) and shoot (no/yes) each get a linear head off the pooled
    vector; loss is the equal-weight sum of the two cross-entropies.
    """

    def __init__(self, config: DoomConvStemConfig):
        super().__init__(config)
        self.turn_head = nn.Linear(self.hidden_size, 3)
        self.shoot_head = nn.Linear(self.hidden_size, 2)
        self.post_init()

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


class DoomConvStemPolicy(DoomConvStemPreTrainedModel):
    """Actor-critic over the conv-stem trunk, on the flat uint8 observation (B, OBS_LEN).

    Built fully trainable; call set_trainable(unfreeze_blocks=N) to freeze all but the
    last N encoder layers + final_norm (and the stem, unless train_stem). The attention
    pool and the three heads are always trainable.
    """

    def __init__(self, config: DoomConvStemConfig):
        super().__init__(config)
        self.turn_head = nn.Linear(self.hidden_size, 3)
        self.shoot_head = nn.Linear(self.hidden_size, 2)
        self.value_head = nn.Linear(self.hidden_size, 1)
        self.set_trainable(unfreeze_blocks=0)
        self.post_init()

    def _init_weights(self, module):
        if module is self.turn_head or module is self.shoot_head or module is self.value_head:
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
        else:
            super()._init_weights(module)

    def set_trainable(self, unfreeze_blocks=0, train_stem=False):
        """unfreeze_blocks=0 trains everything; N>0 trains only the last N encoder layers +
        final_norm, with the stem frozen unless train_stem."""
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

    def _unpack(self, obs):
        obs = obs.to(torch.uint8) if obs.dtype != torch.uint8 else obs
        frames = obs[:, : self._stack_pixels].view(-1, *self._frame_shape)
        prev = obs[:, self._stack_pixels :].long()  # (B, N_PREV)
        return frames, prev

    def forward(self, frames, prev_actions):
        """(frames, prev_actions) -> {"turn_logits", "shoot_logits", "value"}."""
        pooled = self.encode(frames, prev_actions)
        return {
            "turn_logits": self.turn_head(pooled),
            "shoot_logits": self.shoot_head(pooled),
            "value": self.value_head(pooled).squeeze(-1),
        }

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


AutoConfig.register("doom_conv_stem", DoomConvStemConfig)
AutoModel.register(DoomConvStemConfig, DoomConvStemPolicy)
