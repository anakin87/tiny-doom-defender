"""Turn a checkpoint into a playing policy and score it: loading, the episode loop,
and the metrics every script reports."""

import json
import os

import numpy as np
import torch

from tiny_doom_defender.model import ConvStemPolicy

DEFAULT_BASE_MODEL = "models/doom-cnn-4L-no-fwd"


# =============================================================================
# Loading a checkpoint
# =============================================================================


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
    """True for a PPO snapshot (a .pt file holding a full policy state dict); an SFT
    checkpoint is a directory containing model.pt."""
    return os.path.isfile(ckpt) and ckpt.endswith(".pt")


def resolve_base_model(ckpt, base_model=None):
    """Which encoder dir to rebuild the architecture from.

    An explicit base_model wins. Otherwise a PPO snapshot is a bare state dict, so use
    the encoder its run recorded in manifest.json. Nothing catches a wrong one: heads,
    attention window and RoPE thetas change the forward pass without changing any
    parameter shape, so a mismatched encoder loads without error.
    """
    if base_model is not None:
        return base_model
    manifest = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "manifest.json")
    if is_policy_checkpoint(ckpt) and os.path.isfile(manifest):
        with open(manifest) as f:
            from_run = json.load(f).get("base_model")
        if from_run:
            print(f"  base-model from {manifest}: {from_run}")
            return from_run
    return DEFAULT_BASE_MODEL


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
