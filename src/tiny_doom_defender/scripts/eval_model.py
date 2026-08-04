"""
Evaluate a checkpoint IN-GAME: frags over the held-out TEST seeds.

Takes either checkpoint kind, an SFT dir or a PPO snapshot. Episodes are split across
--workers processes, each running its own VizDoom instance.

Usage:
  eval-model --ckpt output/cnn-ppo/policy_best.pt
  eval-model --ckpt output/cnn-sft/best --episodes 100
"""

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from tiny_doom_defender.config import SEED_TEST
from tiny_doom_defender.env import DefendCenterConvStemEnv
from tiny_doom_defender.evaluation import (
    DEFAULT_BASE_MODEL,
    build_policy,
    pick_device,
    play_episodes,
    resolve_base_model,
    summarize,
)


def _worker(payload):
    ckpt, base_model, seeds, mode, device_str, sample_seed = payload
    device = torch.device(device_str)
    policy = build_policy(ckpt, base_model, device, verbose=False)
    env = DefendCenterConvStemEnv()
    try:
        return play_episodes(policy, env, device, seeds, mode, sample_seed=sample_seed)
    finally:
        env.close()


def evaluate(ckpt, base_model, seeds, mode, device_str, sample_seed, workers):
    if workers <= 1:
        device = torch.device(device_str)
        policy = build_policy(ckpt, base_model, device)
        env = DefendCenterConvStemEnv()
        try:
            parts = [play_episodes(policy, env, device, seeds, mode, sample_seed=sample_seed)]
        finally:
            env.close()
        return summarize(parts)

    n = min(workers, len(seeds))
    chunk = (len(seeds) + n - 1) // n
    chunks = [seeds[i : i + chunk] for i in range(0, len(seeds), chunk)]
    payloads = [(ckpt, base_model, c, mode, device_str, sample_seed) for c in chunks]
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
        parts = list(ex.map(_worker, payloads))
    return summarize(parts)


def report(mode, m, elapsed):
    print(f"\n[{mode}] frags/episode over {m['n_episodes']} eps  ({elapsed:.1f}s):")
    print(
        f"  mean={m['mean']:.2f}  std={m['std']:.2f}  min={m['min']:.0f}  max={m['max']:.0f}  median={m['median']:.1f}"
    )
    if "accuracy" in m:
        print(f"  accuracy (kills/bullets)={m['accuracy']:.3f}  bullets/ep={m['bullets']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="output/cnn-ppo/policy_best.pt", help="SFT dir (best/) or a PPO policy .pt file.")
    ap.add_argument(
        "--base-model",
        default=None,
        help=f"Encoder dir the policy architecture is built from. Default: the base_model recorded "
        f"in the run's manifest.json when --ckpt is a PPO snapshot, else {DEFAULT_BASE_MODEL}.",
    )
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-base", type=int, default=SEED_TEST, help="First TEST seed.")
    ap.add_argument("--mode", default="argmax", choices=["argmax", "sampled", "both"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42, help="Base for the per-episode torch reseed in --mode sampled.")
    args = ap.parse_args()

    device = pick_device(args.device, parallel=args.workers > 1)
    seeds = [args.seed_base + i for i in range(args.episodes)]
    print(
        f"Evaluating MODEL {args.ckpt} on defend_the_center: {args.episodes} eps, "
        f"seeds {seeds[0]}..{seeds[-1]} (held-out), device={device.type}, "
        f"workers={args.workers}, mode={args.mode}"
    )
    base_model = resolve_base_model(args.ckpt, args.base_model)

    modes = ["argmax", "sampled"] if args.mode == "both" else [args.mode]
    for mode in modes:
        t0 = time.time()
        m = evaluate(args.ckpt, base_model, seeds, mode, device.type, args.seed, args.workers)
        report(mode, m, time.time() - t0)


if __name__ == "__main__":
    main()
