"""
Watch a trained policy play defend_the_center in a LIVE DOOM window.

What you see is fed to the policy through the same observation pipeline eval-model
scores. Playback is paced to ~real time; the terminal prints HP / kills / ammo, the
chosen action, and the per-axis probabilities behind it.

Usage:
  play-doom --ckpt output/cnn-ppo/policy_best
  play-doom --ckpt output/cnn-ppo/policy_best --seed 10000 --episodes 1 --fps 20 --mode sampled
"""

import argparse
import time
from collections import Counter

import numpy as np
import torch
import vizdoom

from tiny_doom_defender.constants import FRAME_SKIP, TICS_PER_SECOND
from tiny_doom_defender.env import DefendCenterConvStemEnv
from tiny_doom_defender.evaluation import build_policy

TURN_NAMES = ("left", "none", "right")


@torch.no_grad()
def decide(policy, obs, device, mode):
    """One observation -> (turn, shoot, turn_probs, shoot_probs)."""
    turn_p, shoot_p = policy.action_probs(torch.from_numpy(obs[None]).to(device))
    turn_p, shoot_p = turn_p[0].cpu(), shoot_p[0].cpu()
    if mode == "argmax":
        turn, shoot = int(turn_p.argmax()), int(shoot_p.argmax())
    else:
        turn, shoot = int(torch.multinomial(turn_p, 1)), int(torch.multinomial(shoot_p, 1))
    return turn, shoot, turn_p.numpy(), shoot_p.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="output/cnn-ppo/policy_best", help="SFT dir (best/) or a PPO snapshot dir.")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="First episode seed, then seed+1, ... (held-out TEST seeds are 10000+). Omit for random.",
    )
    ap.add_argument("--mode", default="argmax", choices=["argmax", "sampled"])
    ap.add_argument("--fps", type=float, default=35.0, help="Playback pace (35=real time, 20=slow-mo, 60+=fast).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--print-every", type=int, default=10, help="HUD print cadence (steps).")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Loading policy {args.ckpt}...")
    policy = build_policy(args.ckpt, device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params:,} params  |  mode={args.mode}")

    print("\nStarting DOOM (defend_the_center) — a window should open...")
    env = DefendCenterConvStemEnv(visible=True)
    step_interval = FRAME_SKIP / max(args.fps, 1e-6)  # wall-seconds per decision

    frags_hist = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep if args.seed is not None else None
            obs, _ = env.reset(seed=seed)
            seed_str = f"seed {seed}" if seed is not None else "random seed"
            print(f"\n{'=' * 64}\nEpisode {ep + 1}/{args.episodes}  ({seed_str})\n{'=' * 64}")

            step, done, info = 0, False, {}
            counter = Counter()
            while not done:
                t0 = time.perf_counter()
                turn, shoot, turn_p, shoot_p = decide(policy, obs, device, args.mode)
                act_name = f"{TURN_NAMES[turn]}{'+shoot' if shoot else ''}"
                counter[act_name] += 1

                if step % args.print_every == 0:
                    g = env.game
                    print(
                        f"  step {step:4d} | "
                        f"HP={g.get_game_variable(vizdoom.GameVariable.HEALTH):3.0f} "
                        f"K={g.get_game_variable(vizdoom.GameVariable.KILLCOUNT):2.0f} "
                        f"ammo={g.get_game_variable(vizdoom.GameVariable.AMMO2):2.0f} | "
                        f"{act_name:12s} | turn L/N/R={turn_p[0]:.2f}/{turn_p[1]:.2f}/{turn_p[2]:.2f} "
                        f"shoot={shoot_p[1]:.2f}"
                    )

                obs, _, term, trunc, info = env.step(np.array([turn, shoot], dtype=np.int32))
                done = term or trunc
                step += 1

                dt = time.perf_counter() - t0  # pace to ~real time
                if dt < step_interval:
                    time.sleep(step_interval - dt)

            frags = info.get("killcount", float("nan"))
            bullets = info.get("bullets", float("nan"))
            frags_hist.append(frags)
            acc = frags / bullets if bullets else float("nan")
            surv = env.game.get_episode_time() / TICS_PER_SECOND
            print(
                f"\n  --- Episode {ep + 1}: frags={frags:.0f}  bullets={bullets:.0f}  "
                f"acc={acc:.3f}  survived={surv:.1f}s ---"
            )
            for name, c in counter.most_common():
                pct = 100 * c / max(step, 1)
                print(f"    {name:12s}: {c:4d} ({pct:4.1f}%)  {'#' * int(pct / 3)}")
    finally:
        env.close()

    if frags_hist:
        arr = np.array(frags_hist, dtype=float)
        print(f"\n{'=' * 64}\n{len(arr)} episodes: frags mean={arr.mean():.2f} [{arr.min():.0f}, {arr.max():.0f}]")


if __name__ == "__main__":
    main()
