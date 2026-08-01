"""RL layer around the conv-stem network in model.py: the actor-critic policy, the
VizDoom env that feeds it, and the eval loop shared by training and scoring."""

import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import vizdoom

from tiny_doom_defender.config import (
    FRAME_SKIP,
    IN_CH,
    N_FRAMES,
    N_PREV,
    OBS_LEN,
    RES_H,
    RES_W,
    STACK_PIXELS,
    START_ACTION,
)
from tiny_doom_defender.game import screen_to_frame, setup_game
from tiny_doom_defender.model import _StemTrunk
from tiny_doom_defender.utils import combine_action, pack_obs, stack_channels

# =============================================================================
# Policy (actor-critic over the shared conv-stem trunk)
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


def load_sft_into_policy(policy, sft_checkpoint, device, verbose=True):
    """Overlay an SFT checkpoint (a dir with model.pt, or the model.pt path) onto `policy`.

    Every key present in both with a matching shape is copied; the rest keeps its fresh
    init. Trunk, pool and the two action heads match DoomConvStemClassifier by name, so
    value_head is the only thing left untouched.
    """
    sft_path = os.path.join(sft_checkpoint, "model.pt") if os.path.isdir(sft_checkpoint) else sft_checkpoint
    if not os.path.isfile(sft_path):
        raise FileNotFoundError(f"SFT checkpoint not found at {sft_path}")
    sft_state = torch.load(sft_path, map_location="cpu", weights_only=True)
    fresh = policy.state_dict()
    merged = dict(fresh)
    matched = 0
    for k, v in sft_state.items():
        if k in merged and merged[k].shape == v.shape:
            merged[k] = v.to(device)
            matched += 1
    fresh_only = sorted(set(fresh) - set(sft_state))
    if verbose:
        print(f"  SFT keys loaded: {matched}/{len(sft_state)};  random-init kept: {fresh_only}")
    policy.load_state_dict(merged)


def is_policy_checkpoint(ckpt):
    """True for a PPO snapshot (a .pt file holding a full policy state dict); False for an
    SFT checkpoint, which is a directory containing model.pt."""
    return os.path.isfile(ckpt) and ckpt.endswith(".pt")


def build_policy(ckpt, base_model, device, verbose=True):
    """Load either checkpoint kind into a ready-to-score policy, in eval mode.

    Fully unfrozen: unfreeze_blocks only flips requires_grad, which nothing here reads.
    """
    policy = ConvStemPolicy(base_model, unfreeze_blocks=0).to(device)
    if is_policy_checkpoint(ckpt):
        policy.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        load_sft_into_policy(policy, ckpt, device, verbose=verbose)
    policy.eval()
    return policy


# =============================================================================
# Vision-only env
# =============================================================================


class DefendCenterConvStemEnv(gym.Env):
    """defend_the_center with MultiDiscrete([3, 2]) actions and a vision-only obs.

    Holds the rolling buffers behind the (OBS_LEN,) observation: the last N_FRAMES frames
    and the last N_PREV actions. The game comes from game.setup_game, so resolution,
    screen format and button order match the recorder. `info` at episode end carries
    killcount plus the bullets/hits behind the accuracy diagnostic.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self._game = None  # lazy init in reset() so subprocess spawn stays cheap
        self._start_ammo = float("nan")  # AMMO2 at reset, for bullets-fired accounting
        self._frame_buf = deque(maxlen=N_FRAMES)
        self._act_hist = deque(maxlen=N_PREV)

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(OBS_LEN,), dtype=np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([3, 2])

    def _ensure_game(self):
        if self._game is None:
            self._game = setup_game()

    def _prev_list(self):
        # [older ... newer], left-padded with START when history is short.
        h = list(self._act_hist)
        return [START_ACTION] * (N_PREV - len(h)) + h

    def _obs(self):
        return pack_obs(stack_channels(self._frame_buf), self._prev_list())

    def _push_frame(self, frame):
        # Keep the buffer at exactly N_FRAMES: when short (fresh episode) fill it by
        # repeating this frame — the same rule the recorded dataset uses at episode start.
        if len(self._frame_buf) < N_FRAMES:
            self._frame_buf.clear()
            for _ in range(N_FRAMES):
                self._frame_buf.append(frame)
        else:
            self._frame_buf.append(frame)

    def _terminal_obs(self):
        return np.zeros(OBS_LEN, dtype=np.uint8)

    def _episode_info(self):
        info = {}
        try:
            info["killcount"] = float(self._game.get_game_variable(vizdoom.GameVariable.KILLCOUNT))
        except Exception:
            info["killcount"] = float("nan")
        # Accuracy accounting: bullets fired = start AMMO2 - final AMMO2, hits = HITCOUNT.
        try:
            info["bullets"] = self._start_ammo - float(self._game.get_game_variable(vizdoom.GameVariable.AMMO2))
            info["hits"] = float(self._game.get_game_variable(vizdoom.GameVariable.HITCOUNT))
        except Exception:
            info["bullets"] = float("nan")
            info["hits"] = float("nan")
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_game()
        if seed is not None:
            self._game.set_seed(int(seed) % (2**31 - 1))
        self._game.new_episode()
        self._act_hist.clear()
        self._frame_buf.clear()  # fresh episode: no carry-over frames
        try:
            self._start_ammo = float(self._game.get_game_variable(vizdoom.GameVariable.AMMO2))
        except Exception:
            self._start_ammo = float("nan")
        state = self._game.get_state()
        if state is None:
            return self._terminal_obs(), {}
        self._push_frame(screen_to_frame(state.screen_buffer))  # -> N_FRAMES copies
        return self._obs(), {}

    def step(self, action):
        turn_a, shoot_a = int(action[0]), int(action[1])
        buttons = [shoot_a == 1, turn_a == 0, turn_a == 2]  # [ATTACK, TURN_LEFT, TURN_RIGHT]
        reward = float(self._game.make_action(buttons, FRAME_SKIP))
        self._act_hist.append(combine_action(turn_a, shoot_a))
        if self._game.is_episode_finished():
            return self._terminal_obs(), reward, True, False, self._episode_info()
        state = self._game.get_state()
        if state is None:
            return self._terminal_obs(), reward, True, False, self._episode_info()
        self._push_frame(screen_to_frame(state.screen_buffer))
        return self._obs(), reward, False, False, {}

    def close(self):
        if self._game is not None:
            try:
                self._game.close()
            except Exception:
                pass
            self._game = None


def make_env():
    """Thunk for gymnasium.vector.AsyncVectorEnv (PPO rollout collection)."""

    def _thunk():
        return DefendCenterConvStemEnv()

    return _thunk


# =============================================================================
# Shared eval loop
# =============================================================================


@torch.no_grad()
def play_episodes(policy, env, device, seeds, mode, sample_seed=0):
    """Drive `env` with `policy` over `seeds` -> raw per-episode frags/bullets/hits.

    mode 'argmax' is greedy; 'sampled' draws per axis, reseeding torch to sample_seed+seed
    each episode so splitting a run across N workers gives the same scores.
    """
    frags, bullets, hits = [], [], []
    for s in seeds:
        if mode == "sampled":
            torch.manual_seed(sample_seed + s)
        obs, _ = env.reset(seed=s)
        done = False
        kc, bl, ht = float("nan"), float("nan"), float("nan")
        while not done:
            obs_t = torch.from_numpy(obs[None]).to(device)
            if mode == "argmax":
                a = policy.get_argmax_action(obs_t)[0].cpu().numpy()
            else:
                a, _, _, _ = policy.get_action_and_value(obs_t)
                a = a[0].cpu().numpy()
            obs, r, term, trunc, info = env.step(a.astype(np.int32))
            done = term or trunc
            if done:
                kc = info.get("killcount", float("nan"))
                bl = info.get("bullets", float("nan"))
                ht = info.get("hits", float("nan"))
        frags.append(kc)
        bullets.append(bl)
        hits.append(ht)
    return {"frags": frags, "bullets": bullets, "hits": hits}


def summarize(parts):
    """Merge `play_episodes` pieces (one per worker) into the reported metrics.

    mean frags = accuracy x mean bullets fired, so accuracy and bullets/ep decompose it.
    """
    frags, bullets, hits = [], [], []
    for p in parts:
        frags += list(p["frags"])
        bullets += list(p["bullets"])
        hits += list(p["hits"])
    f = np.array(frags, dtype=float)
    out = {
        "n_episodes": len(frags),
        "mean": float(np.nanmean(f)),
        "std": float(np.nanstd(f)),
        "min": float(np.nanmin(f)),
        "max": float(np.nanmax(f)),
        "median": float(np.nanmedian(f)),
    }

    bl = np.array(bullets, dtype=float)
    if bl.size and np.isfinite(bl).any():
        bl_sum = float(np.nansum(bl))
        out["bullets"] = float(np.nanmean(bl))
        out["accuracy"] = (float(np.nansum(f)) / bl_sum) if bl_sum > 0 else float("nan")
    return out


def fmt_summary(tag, m):
    """One-line rendering of a `summarize` dict."""
    acc = f"  acc={m['accuracy']:.3f} bullets={m['bullets']:.2f}" if "accuracy" in m else ""
    return (
        f"  [{tag}] frags mean={m['mean']:.2f} ± {m['std']:.2f}  "
        f"[{m['min']:.0f}, {m['max']:.0f}] median={m['median']:.1f}{acc}"
    )


def pick_device(device_arg="auto", parallel=False):
    """'auto' -> cuda/mps/cpu, but cpu when parallel: worker processes would contend on the
    single accelerator. Pinning the device also keeps scores comparable across runs (MPS
    and CPU differ in float and RNG)."""
    if device_arg != "auto":
        return torch.device(device_arg)
    if parallel:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
