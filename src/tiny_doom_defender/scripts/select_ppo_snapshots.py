"""
Rank the snapshots of a PPO run offline and keep the best one.

Rebuilds the policy on the encoder recorded in the run's manifest.json, replays every
snapshot on the SEED_SELECTION pool, ranks by mean frags and copies the winner to
<output-dir>/policy_best.pt.

Usage:
  select-ppo-snapshots --output-dir output/cnn-ppo
  select-ppo-snapshots --output-dir output/cnn-ppo --iters 20-30
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import torch

from tiny_doom_defender.config import SEED_SELECTION
from tiny_doom_defender.env import DefendCenterConvStemEnv
from tiny_doom_defender.evaluation import build_policy, fmt_summary, pick_device, play_episodes, summarize


def parse_iters(spec):
    """'20-30,35' -> [20, ..., 30, 35] (order preserved, duplicates dropped)."""
    if not spec:
        return None
    out, seen = [], set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            rng = range(int(a), int(b) + 1)
        else:
            rng = [int(part)]
        for v in rng:
            if v not in seen:
                out.append(v)
                seen.add(v)
    return out


def eval_snapshots(base_model, output_dir, snaps, seeds, mode, device_str, sample_seed, progress=False):
    """Score `snaps` sequentially, reusing one VizDoom instance."""
    device = torch.device(device_str)
    env = DefendCenterConvStemEnv()
    out = []
    try:
        for s in snaps:
            snap_path = os.path.join(output_dir, s["path"])
            if not os.path.isfile(snap_path):
                continue
            policy = build_policy(snap_path, base_model, device)
            m = summarize([play_episodes(policy, env, device, seeds, mode, sample_seed=sample_seed)])
            out.append((s["iter"], s["path"], m))
            if progress:
                print(fmt_summary(f"iter {s['iter']}", m), flush=True)
    finally:
        env.close()
    return out


def _worker(payload):
    (base_model, output_dir, snaps, seeds, mode, device_str, sample_seed) = payload
    # progress=True: workers share the parent's stdout, so scores stream in as they finish.
    return eval_snapshots(base_model, output_dir, snaps, seeds, mode, device_str, sample_seed, progress=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, help="A train_ppo.py --output dir (must contain manifest.json).")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed-base", type=int, default=SEED_SELECTION)
    p.add_argument("--iters", default=None, help="Iterations/ranges to evaluate, e.g. '20-30,35'. Default: all.")
    p.add_argument("--mode", default="argmax", choices=["argmax", "sampled"])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--save-best", action="store_true", default=True)
    p.add_argument("--no-save-best", dest="save_best", action="store_false")
    p.add_argument("--report", default=None, help="Where to write the JSON report. Default: inside --output-dir.")
    args = p.parse_args()

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        sys.exit(f"ERROR: no manifest.json at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    base_model = manifest["base_model"]
    print(f"Loaded manifest: {len(manifest['snapshots'])} snapshots  (base_model={base_model})")

    by_iter = {s["iter"]: s for s in manifest["snapshots"]}
    order = parse_iters(args.iters)
    if order is not None:
        snapshots = [by_iter[i] for i in order if i in by_iter]
    else:
        snapshots = sorted(manifest["snapshots"], key=lambda s: s["iter"])
    if not snapshots:
        sys.exit("ERROR: no snapshots selected.")

    device = pick_device(args.device, parallel=args.workers > 1)
    seeds = [args.seed_base + i for i in range(args.episodes)]
    print(
        f"\n=== Re-evaluating {len(snapshots)} snapshots "
        f"({args.episodes} eps, seed_base={args.seed_base}, mode={args.mode}, "
        f"device={device.type}, workers={args.workers}) ==="
    )

    t0 = time.time()
    if args.workers <= 1:
        flat = eval_snapshots(
            base_model, args.output_dir, snapshots, seeds, args.mode, device.type, args.seed_base, progress=True
        )
    else:
        n = min(args.workers, len(snapshots))
        chunks = [snapshots[i::n] for i in range(n)]
        payloads = [(base_model, args.output_dir, c, seeds, args.mode, device.type, args.seed_base) for c in chunks]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
            flat = [r for part in ex.map(_worker, payloads) for r in part]

    results = [
        {"iter": it, "path": path, "score": m["mean"], "std": m["std"], "min": m["min"], "max": m["max"], "metrics": m}
        for (it, path, m) in flat
    ]
    if not results:
        sys.exit("ERROR: no snapshots evaluated.")
    results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n=== Ranked (best first, {time.time() - t0:.0f}s) ===")
    for r in results:
        print(fmt_summary(f"iter {r['iter']}", r["metrics"]))

    winner = results[0]
    print(f"\nWinner: iter {winner['iter']}  frags={winner['score']:.2f} ± {winner['std']:.2f}")

    if args.save_best:
        best_path = os.path.join(args.output_dir, "policy_best.pt")
        sd = torch.load(os.path.join(args.output_dir, winner["path"]), map_location="cpu", weights_only=True)
        torch.save(sd, best_path)
        print(f"Saved winning snapshot -> {best_path}")

    report_path = args.report or os.path.join(args.output_dir, f"rerank_seed{args.seed_base}_{args.mode}.json")
    with open(report_path, "w") as f:
        json.dump(
            {
                "manifest": manifest,
                "episodes": args.episodes,
                "seed_base": args.seed_base,
                "mode": args.mode,
                "winner_iter": winner["iter"],
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Report written to {report_path}")

    print("\nNB: biased selection score. Report the TEST number:")
    print(f"  eval-model --ckpt {args.output_dir}/policy_best.pt --episodes 100")


if __name__ == "__main__":
    main()
