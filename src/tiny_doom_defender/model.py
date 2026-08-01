"""
Conv-stem "eye" in front of ModernBERT: the stem turns a frame stack into a fixed
1000-cell grid fed via `inputs_embeds`. Every cell is a real patch (no padding,
unlike a tokenized sequence), so the attention mask is all-ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from tiny_doom_defender.config import IN_CH, N_ACTION_STATES, N_PREV
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
