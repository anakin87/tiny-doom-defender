"""
PPO-refine the conv-stem SFT policy on defend_the_center (vision-only).

CleanRL-style PPO: GAE, clipped surrogate, advantage normalization, KL early-stop,
two-LR AdamW. Policy and env come from ppo_core.py; rollouts are collected from
--num-envs VizDoom processes behind a spawned AsyncVectorEnv.

Every iteration writes a snapshot + manifest.json, so a crash keeps its progress. The
last iterate is not necessarily the best one — pick which to keep offline with
select-ppo-snapshots.

Usage:
    train-ppo --sft-checkpoint output/cnn-sft/best --output output/cnn-ppo
"""

# PYTORCH_MPS_HIGH_WATERMARK_RATIO must be set before torch is imported (the uint8
# rollout buffer is large), hence the write between the imports.
# ruff: noqa: E402
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import contextlib
import json
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode

from tiny_doom_defender.config import OBS_LEN, SEED_ROLLOUT, SEED_SELECTION
from tiny_doom_defender.ppo_core import ConvStemPolicy, load_sft_into_policy, make_env, pick_device


def amp_ctx(device, bf16):
    """bf16 autocast on CUDA; a no-op on MPS and CPU."""
    if bf16 and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def write_manifest(args, snapshots):
    """Snapshot index + run provenance. base_model is the field read back: the selector
    and eval_model rebuild the architecture from it."""
    with open(os.path.join(args.output, "manifest.json"), "w") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "sft_checkpoint": args.sft_checkpoint,
                "unfreeze_blocks": args.unfreeze_blocks,
                "train_stem": args.train_stem,
                "snapshots": [{"iter": s["iter"], "path": s["path"]} for s in snapshots],
            },
            f,
            indent=2,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sft-checkpoint", default="output/cnn-sft/best", help="SFT dir (with model.pt) or a model.pt path."
    )
    p.add_argument(
        "--base-model", default="models/doom-cnn-4L-no-fwd", help="Encoder dir the policy architecture is built from."
    )
    p.add_argument("--output", default="output/cnn-ppo")

    p.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=1,
        help="0 = fully unfrozen. N>0 = last N encoder layers + final_norm only.",
    )
    p.add_argument(
        "--train-stem",
        action="store_true",
        help="Also fine-tune the conv stem at the encoder LR. Default: stem frozen when --unfreeze-blocks>0.",
    )

    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument(
        "--num-steps",
        type=int,
        default=256,
        help="Steps per env per rollout. batch = num_envs * num_steps, and the on-device "
        "uint8 obs buffer is batch * OBS_LEN bytes.",
    )
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--num-minibatches", type=int, default=16)

    p.add_argument("--lr", type=float, default=5e-4, help="Heads LR.")
    p.add_argument("--encoder-lr", type=float, default=5e-5, help="Encoder (+ stem, if --train-stem) LR.")
    p.add_argument("--bf16", action="store_true", default=True, help="Autocast the forward pass (CUDA only).")
    p.add_argument("--no-bf16", dest="bf16", action="store_false")

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.03)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05, help="Stop the update epochs once approx KL exceeds this.")

    p.add_argument("--time-budget-s", type=int, default=5400, help="Stop after this much wall-clock training time.")
    p.add_argument("--max-iterations", type=int, default=30)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=SEED_ROLLOUT)
    args = p.parse_args()

    device = pick_device(args.device, parallel=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    unfreeze_desc = (
        "fully unfrozen"
        if args.unfreeze_blocks == 0
        else f"last {args.unfreeze_blocks} block(s) + final_norm{' + stem' if args.train_stem else ' (stem frozen)'}"
    )
    print(f"=== Phase 1: Build policy ({unfreeze_desc})  [device={device}] ===")
    policy = ConvStemPolicy(args.base_model, unfreeze_blocks=args.unfreeze_blocks, train_stem=args.train_stem).to(
        device
    )
    print(f"  SFT ckpt: {args.sft_checkpoint}")
    load_sft_into_policy(policy, args.sft_checkpoint, device)
    n_total = sum(q.numel() for q in policy.parameters())
    n_train = sum(q.numel() for q in policy.parameters() if q.requires_grad)
    print(f"  Trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.1f}%)")

    print(f"\n=== Phase 2: Spawning {args.num_envs} VizDoom envs ===")
    envs = gym.vector.AsyncVectorEnv(
        [make_env() for _ in range(args.num_envs)],
        context="spawn",
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": policy.encoder_trainable_params(), "lr": args.encoder_lr},
            {"params": policy.head_trainable_params(), "lr": args.lr},
        ],
        eps=1e-5,
    )
    print(f"  Optimizer: AdamW two-LR (encoder={args.encoder_lr}, heads={args.lr})")

    print(
        f"\n=== Phase 3: PPO (budget {args.time_budget_s}s, "
        f"autocast={'bf16' if args.bf16 and device.type == 'cuda' else 'off'}) ==="
    )
    obs_buf = torch.zeros(args.num_steps, args.num_envs, OBS_LEN, dtype=torch.uint8, device=device)
    actions_buf = torch.zeros(args.num_steps, args.num_envs, 2, dtype=torch.long, device=device)
    logprobs_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    rewards_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    dones_buf = torch.zeros(args.num_steps, args.num_envs, device=device)
    values_buf = torch.zeros(args.num_steps, args.num_envs, device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.from_numpy(np.asarray(next_obs_np)).to(torch.uint8).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    train_start = time.time()
    global_step = 0
    iteration = 0
    ret_window = deque(maxlen=50)
    ep_ret_acc = np.zeros(args.num_envs, dtype=np.float64)
    snapshots = []

    while True:
        elapsed = time.time() - train_start
        if elapsed > args.time_budget_s:
            print(f"\n  Time budget reached after {iteration} iterations.")
            break
        if iteration >= args.max_iterations:
            print("\n  Max iterations reached.")
            break
        iteration += 1
        it_start = time.time()

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad(), amp_ctx(device, args.bf16):
                action, logprob, _, value = policy.get_action_and_value(next_obs)
            actions_buf[step] = action
            logprobs_buf[step] = logprob.float()
            values_buf[step] = value.float()

            action_np = action.cpu().numpy().astype(np.int32)
            next_obs_np, reward, term, trunc, infos = envs.step(action_np)
            done = np.logical_or(term, trunc)
            rewards_buf[step] = torch.from_numpy(reward).float().to(device)
            next_obs = torch.from_numpy(np.asarray(next_obs_np)).to(torch.uint8).to(device)
            next_done = torch.from_numpy(done.astype(np.float32)).to(device)
            ep_ret_acc += reward
            for j in range(args.num_envs):
                if done[j]:
                    ret_window.append(float(ep_ret_acc[j]))
                    ep_ret_acc[j] = 0.0

        with torch.no_grad(), amp_ctx(device, args.bf16):
            next_value = policy.get_value(next_obs).float()
        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones_buf[t + 1]
                nextvalues = values_buf[t + 1]
            delta = rewards_buf[t] + args.gamma * nextvalues * nextnonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values_buf

        b_obs = obs_buf.reshape(-1, OBS_LEN)
        b_actions = actions_buf.reshape(-1, 2)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        batch_size = args.num_envs * args.num_steps
        minibatch_size = batch_size // args.num_minibatches

        b_inds = np.arange(batch_size)
        clipfracs, pg_losses, v_losses, ent_losses = [], [], [], []
        approx_kl = 0.0
        epochs_run = 0
        kl_early_stopped = False
        grad_norm = torch.tensor(0.0)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                with amp_ctx(device, args.bf16):
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        b_obs[mb_inds], action=b_actions[mb_inds]
                    )
                logratio = newlogprob.float() - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)  # normalized per minibatch
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue.float() - b_returns[mb_inds]) ** 2).mean()
                ent_loss = entropy.float().mean()

                loss = pg_loss - args.ent_coef * ent_loss + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    [q for q in policy.parameters() if q.requires_grad], args.max_grad_norm
                )
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
            epochs_run += 1
            if args.target_kl is not None and approx_kl > args.target_kl:
                kl_early_stopped = True
                break

        dt = time.time() - it_start
        recent_mean = float(np.mean(ret_window)) if ret_window else float("nan")

        snap_path = os.path.join(args.output, f"policy_iter{iteration}.pt")
        torch.save({k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}, snap_path)
        snapshots.append({"iter": iteration, "path": os.path.basename(snap_path)})
        write_manifest(args, snapshots)

        kl_marker = "!" if kl_early_stopped else " "
        print(
            f"it {iteration:4d} ({dt:5.1f}s, total {elapsed:6.1f}s, steps={global_step:7d})  "
            f"recent_ep_ret={recent_mean:5.2f} (n={len(ret_window)})  "
            f"pg={np.mean(pg_losses):+.4f}  v={np.mean(v_losses):.4f}  "
            f"ent={np.mean(ent_losses):.3f}  kl={approx_kl:.4f}{kl_marker} "
            f"clip={np.mean(clipfracs):.3f}  gn={float(grad_norm):.3f}  "
            f"ep={epochs_run}/{args.update_epochs}",
            flush=True,
        )

    try:
        envs.close()
    except Exception:
        pass

    print("\n=== Training done ===")
    print(f"  iterations: {iteration}   snapshots: {len(snapshots)}")
    print(f"\nNext: pick the best snapshot offline (selection seeds from {SEED_SELECTION}):")
    print(f"  select-ppo-snapshots --output-dir {args.output}")


if __name__ == "__main__":
    main()
