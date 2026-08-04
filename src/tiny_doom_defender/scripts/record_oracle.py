"""
Hardcoded oracle teacher + dataset recorder for the conv-stem model.

The oracle plays defend_the_center off the labels buffer (privileged, but every
decision is a function of what's on screen, so it stays pixel-learnable):
  - patrol: rotate in one fixed direction per episode so enemies can't pile up
    behind us;
  - fire: whenever an enemy body, expanded by EDGE_FRAC per side, is under the
    crosshair — hold aim (pause the sweep) while firing, else keep sweeping.
Actions are turn x shoot (the player never moves).

Writes a streaming dataset that train_sft.py consumes unchanged (frames never all
in RAM at once):
  <output>/frames.u8         raw uint8, N * RES_H*RES_W*3 bytes, C-order, record order
  <output>/labels.npz        turn, shoot, prev0, prev1, ep  (int arrays, len N)
  <output>/record_meta.json  provenance + frag/survival metrics + shapes

Optionally push the recorded dir to the HuggingFace Hub with
`--save-to-dataset <user>/<repo>` (needs a write token); pass that repo id to
`train-sft --data` to reload it.

Usage:
    record-oracle --episodes 400 --output data/cnn-oracle
    record-oracle --episodes 400 --output data/cnn-oracle --save-to-dataset <user>/<repo>
"""

import argparse
import json
import os
from collections import deque

import numpy as np
import vizdoom

from tiny_doom_defender.constants import (
    FRAME_SKIP,
    N_PREV,
    RES_H,
    RES_W,
    START_ACTION,
    TICS_PER_SECOND,
)
from tiny_doom_defender.game import screen_to_frame, setup_game
from tiny_doom_defender.utils import combine_action

# Screen geometry (RES_640X480). The pistol fires straight ahead, so the crosshair
# is the screen's horizontal center.
SCREEN_W = 640
CENTER_X = SCREEN_W / 2
# Labels that are NOT enemies: the agent's own body/viewmodel and hit effects.
NON_ENEMY = {"DoomPlayer", "BulletPuff", "Blood", "Puff"}
EDGE_FRAC = -0.10  # expand each enemy body 10% per side


def oracle_action(labels, scan_dir):
    """Oracle policy. Returns (turn, shoot, buttons).

    turn 0/1/2 = left/none/right, shoot 0/1.
    buttons = [ATTACK, TURN_LEFT, TURN_RIGHT] for make_action.
    """

    def under_crosshair(lbl):
        inset = lbl.width * EDGE_FRAC  # negative -> expands the box
        return (lbl.x + inset) <= CENTER_X <= (lbl.x + lbl.width - inset)

    shoot = int(any(lbl.object_name not in NON_ENEMY and under_crosshair(lbl) for lbl in labels))
    if shoot:  # pause the sweep, hold aim, fire
        return 1, 1, [True, False, False]
    return scan_dir, 0, [False, scan_dir == 0, scan_dir == 2]


def push_to_hub(data_dir, repo_id, private, commit_message):
    """Push a recorded dataset dir (frames.u8 + labels.npz + record_meta.json) to the
    HuggingFace Hub as raw files. Reload later with `train-sft --data <repo_id>`.
    Requires a WRITE token (huggingface-cli login or HF_TOKEN)."""
    frames_path = os.path.join(data_dir, "frames.u8")
    print(
        f"\nPushing {data_dir}  (frames.u8 = {os.path.getsize(frames_path) / 1e9:.2f} GB) "
        f"-> {repo_id} (private={private})..."
    )
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print("  Repo ready. Uploading (LFS handles frames.u8 — this can take a while)...")
    HfApi().upload_folder(folder_path=data_dir, repo_id=repo_id, repo_type="dataset", commit_message=commit_message)
    print(f"Done: https://huggingface.co/datasets/{repo_id}")
    print(f"Reload it later with:\n  train-sft --data {repo_id} --base-model models/doom-cnn-4L-no-fwd --bf16")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument(
        "--max-frames", type=int, default=0, help="stop after this many frames (0 = no cap, use --episodes)"
    )
    ap.add_argument("--output", default="data/cnn-oracle")
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    ap.add_argument(
        "--seed-base", type=int, default=0, help="episode e uses seed base+e; keep well below 10000 (eval)."
    )
    ap.add_argument(
        "--save-to-dataset",
        default=None,
        metavar="REPO_ID",
        help="After recording, push <output>/ to this HuggingFace Hub dataset repo id "
        "(e.g. <user>/<repo>): raw frames.u8 + labels.npz + record_meta.json. "
        "Needs a WRITE token (huggingface-cli login or HF_TOKEN).",
    )
    ap.add_argument("--private", action="store_true", help="With --save-to-dataset: create the Hub repo private.")
    ap.add_argument(
        "--commit-message",
        default="Upload conv-stem oracle dataset",
        help="With --save-to-dataset: commit message for the upload.",
    )
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    frames_path = os.path.join(args.output, "frames.u8")
    frames_file = open(frames_path, "wb")

    game = setup_game(args.scenario)

    all_turn, all_shoot = [], []
    all_prev0, all_prev1, all_ep = [], [], []
    ep_kills, ep_frames, ep_survival_tics, ep_dead = [], [], [], []
    n = 0

    print(f"Recording {args.scenario} (headless, RGB {RES_H}x{RES_W})...")
    for ep in range(args.episodes):
        game.set_seed(args.seed_base + ep)
        game.new_episode()
        scan_dir = 2 if ep % 2 == 0 else 0  # even=right, odd=left (de-bias turn labels)
        act_hist = deque(maxlen=N_PREV)
        ep_start = n
        reached_cap = False
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break
            if game.get_game_variable(vizdoom.GameVariable.AMMO2) <= 0:
                break  # stop recording at ammo-out; still played out below
            screen, labels = state.screen_buffer, state.labels

            turn, shoot, buttons = oracle_action(labels, scan_dir)

            # prev actions from history BEFORE pushing the current action:
            # prev1 = a_{t-1} (newest), prev0 = a_{t-2} (older); START if absent.
            prev1 = act_hist[-1] if len(act_hist) >= 1 else START_ACTION
            prev0 = act_hist[-2] if len(act_hist) >= 2 else START_ACTION

            frame = screen_to_frame(screen)
            frames_file.write(frame.tobytes())
            all_turn.append(turn)
            all_shoot.append(shoot)
            all_prev0.append(prev0)
            all_prev1.append(prev1)
            all_ep.append(ep)
            n += 1

            act_hist.append(combine_action(turn, shoot))
            game.make_action(buttons, args.frame_skip)
            if args.max_frames and n >= args.max_frames:
                reached_cap = True
                break

        while not game.is_episode_finished():  # play out (unrecorded) for true survival
            st = game.get_state()
            if st is None:
                break
            _, _, buttons = oracle_action(st.labels, scan_dir)
            game.make_action(buttons, args.frame_skip)

        try:
            kills = game.get_game_variable(vizdoom.GameVariable.KILLCOUNT)
        except Exception:
            kills = float("nan")
        dead = bool(game.is_player_dead())
        try:
            survival_tics = int(game.get_episode_time())
        except Exception:
            survival_tics = (n - ep_start) * args.frame_skip
        ep_kills.append(kills)
        ep_frames.append(n - ep_start)
        ep_survival_tics.append(survival_tics)
        ep_dead.append(dead)
        print(
            f"  ep {ep + 1}/{args.episodes}: kills={kills:.0f}  "
            f"survived={survival_tics / TICS_PER_SECOND:4.1f}s  "
            f"{'died' if dead else 'timeout'}  total_frames={n:,}"
        )
        if reached_cap:
            break

    game.close()
    frames_file.close()

    print(f"\nRecorded {n:,} frames over {len(ep_kills)} episodes")
    if n == 0:
        print("No frames; nothing saved.")
        os.remove(frames_path)
        return

    turn_arr = np.array(all_turn, dtype=np.int64)
    shoot_arr = np.array(all_shoot, dtype=np.int64)
    np.savez(
        os.path.join(args.output, "labels.npz"),
        turn=turn_arr,
        shoot=shoot_arr,
        prev0=np.array(all_prev0, dtype=np.int64),
        prev1=np.array(all_prev1, dtype=np.int64),
        ep=np.array(all_ep, dtype=np.int64),
    )

    kk = np.array(ep_kills, dtype=float)
    surv_s = np.array(ep_survival_tics, dtype=float) / TICS_PER_SECOND
    dead_arr = np.array(ep_dead, dtype=bool)
    tc = np.bincount(turn_arr, minlength=3)
    print(
        f"\noracle frags/episode:  mean={np.nanmean(kk):.2f}  median={np.nanmedian(kk):.1f}  "
        f"std={np.nanstd(kk):.2f}  min={np.nanmin(kk):.0f}  max={np.nanmax(kk):.0f}"
    )
    print(f"survival/episode:      mean={surv_s.mean():.1f}s  died {int(dead_arr.sum())}/{len(dead_arr)}")
    print("Per-axis label distribution:")
    print(f"  turn  L/none/R = {tc[0] / n:.3f}/{tc[1] / n:.3f}/{tc[2] / n:.3f}")
    print(f"  shoot no/yes   = {1 - shoot_arr.mean():.3f}/{shoot_arr.mean():.3f}")

    meta = {
        "config": {
            "scenario": args.scenario,
            "frame_skip": args.frame_skip,
            "teacher": "hardcoded sweep-and-hold oracle (turn x shoot, no movement)",
            "edge_frac": EDGE_FRAC,
            "seed_base": args.seed_base,
            "episodes_requested": args.episodes,
            "max_frames": args.max_frames,
            "screen_resolution": "RES_640X480",
            "frame_resolution": [RES_H, RES_W],
            "observation": "vision-only RGB frames (no entities, no depth)",
            "recording_cutoff": "out-of-ammo (post-ammo tail played out but not recorded)",
        },
        "dataset": {
            "total_frames": n,
            "episodes": len(ep_kills),
            "frames_file": "frames.u8",
            "frame_dtype": "uint8",
            "frame_shape": [RES_H, RES_W, 3],
            "frame_bytes": RES_H * RES_W * 3,
            "labels_file": "labels.npz",
            "label_columns": ["turn", "shoot", "prev0", "prev1", "ep"],
        },
        "frags": {
            "mean": float(np.nanmean(kk)),
            "median": float(np.nanmedian(kk)),
            "std": float(np.nanstd(kk)),
            "min": float(np.nanmin(kk)),
            "max": float(np.nanmax(kk)),
        },
        "survival_seconds": {"mean": float(surv_s.mean()), "median": float(np.median(surv_s))},
        "outcome": {"deaths": int(dead_arr.sum()), "death_rate": float(dead_arr.mean())},
        "label_distribution": {
            "turn_L": tc[0] / n,
            "turn_N": tc[1] / n,
            "turn_R": tc[2] / n,
            "shoot_yes": float(shoot_arr.mean()),
        },
    }
    with open(os.path.join(args.output, "record_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {n:,} frames to {frames_path} ({os.path.getsize(frames_path) / 1e6:.0f} MB)")
    print(f"Saved labels + metadata to {args.output}")

    if args.save_to_dataset:
        push_to_hub(args.output, args.save_to_dataset, args.private, args.commit_message)


if __name__ == "__main__":
    main()
