"""Turn a checkpoint into a playing policy and score it: loading, the episode loop,
and the metrics every script reports."""

import numpy as np
import torch

from tiny_doom_defender.modeling_doom import DoomConvStemPolicy
from tiny_doom_defender.utils import check_pipeline_config

# =============================================================================
# Loading a checkpoint
# =============================================================================


def build_policy(ckpt, device):
    """Load a checkpoint dir into a ready-to-score policy, in eval mode.

    Works on either checkpoint kind: a PPO snapshot loads fully; an SFT checkpoint
    is missing value_head, which from_pretrained leaves at its fresh init (eval
    never reads it).
    """
    policy = DoomConvStemPolicy.from_pretrained(ckpt)
    check_pipeline_config(policy.config)
    policy.to(device)
    policy.eval()
    return policy


# =============================================================================
# Playing and scoring
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
