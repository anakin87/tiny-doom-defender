"""
SFT the conv-stem classifier on oracle demonstrations (behavior cloning).

Builds DoomConvStemClassifier (architecture in model.py) from the --base-model
encoder dir and trains it end-to-end. Loss is the equal-weight sum of the turn and
shoot cross-entropies; head biases init to log class priors, which removes the
cold-start majority-class attractor.

The frame-stacking dataset (`ConvStemFrameDataset`) memory-maps `frames.u8` and,
per index, builds the 9-channel stack [F_{t-2}, F_{t-1}, F_t] with episode-boundary
clamping (at episode start the current frame is repeated, so no motion leaks across
episodes). prev0/prev1 come from the label columns. Frames stay uint8; the stem
casts + normalizes on device.

Saves the full state dict as model.pt to <output>/best and <output>/final.

Usage:
    train-sft --data data/cnn-oracle --base-model models/doom-cnn-4L-no-fwd --bf16
"""

import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tiny_doom_defender.config import RES_H, RES_W
from tiny_doom_defender.model import DoomConvStemClassifier

# =============================================================================
# Frame-stacking dataset
# =============================================================================


class ConvStemFrameDataset(Dataset):
    REQUIRED = ("turn", "shoot", "prev0", "prev1", "ep")

    def __init__(self, data_path):
        data_dir = self._resolve(data_path)
        frames_path = os.path.join(data_dir, "frames.u8")
        labels_path = os.path.join(data_dir, "labels.npz")
        if not (os.path.isfile(frames_path) and os.path.isfile(labels_path)):
            raise FileNotFoundError(f"{data_dir} must contain frames.u8 + labels.npz (the recorded oracle dataset)")

        lab = np.load(labels_path)
        missing = [c for c in self.REQUIRED if c not in lab.files]
        if missing:
            raise ValueError(f"labels.npz missing columns {missing}; found {lab.files}")
        self.turn = lab["turn"]
        self.shoot = lab["shoot"]
        self.prev0 = lab["prev0"]
        self.prev1 = lab["prev1"]
        self.ep = lab["ep"]

        n = len(self.turn)
        expected = n * RES_H * RES_W * 3
        actual = os.path.getsize(frames_path)
        if actual != expected:
            raise ValueError(
                f"frames.u8 size {actual} != expected {expected} "
                f"({n} frames x {RES_H}x{RES_W}x3). Corrupt, or recorded at a different "
                f"config.py geometry than the current one."
            )
        self.frames = np.memmap(frames_path, dtype=np.uint8, mode="r", shape=(n, RES_H, RES_W, 3))

        # Predecessor indices with per-episode clamping: p1 = one step back (same
        # episode, else self), p2 = two steps back (same episode, else clamp to p1).
        # Episodes are contiguous blocks, so "i-k same episode" == ep[i-k]==ep[i].
        idx = np.arange(n)
        prev1_same = np.zeros(n, dtype=bool)
        prev1_same[1:] = self.ep[1:] == self.ep[:-1]
        p1 = np.where(prev1_same, idx - 1, idx)
        prev2_same = np.zeros(n, dtype=bool)
        prev2_same[2:] = self.ep[2:] == self.ep[:-2]
        p2 = np.where(prev1_same & prev2_same, idx - 2, p1)
        self._p1, self._p2 = p1, p2
        self._n = n

    @staticmethod
    def _resolve(data_path):
        """Local dir -> use as-is; otherwise treat `data_path` as a HuggingFace Hub
        dataset repo id and snapshot_download it (cached on subsequent runs). The
        snapshot has the same frames.u8 + labels.npz layout."""
        if os.path.isdir(data_path):
            return data_path
        from huggingface_hub import snapshot_download

        print(f"  Downloading HF dataset {data_path} (cached on subsequent runs)...")
        return snapshot_download(repo_id=data_path, repo_type="dataset")

    def __len__(self):
        return self._n

    def label_distribution(self):
        return {
            "turn_L": float((self.turn == 0).mean()),
            "turn_N": float((self.turn == 1).mean()),
            "turn_R": float((self.turn == 2).mean()),
            "shoot": float((self.shoot == 1).mean()),
        }

    def __getitem__(self, i):
        f2 = self.frames[self._p2[i]]  # oldest  (F_{t-2})
        f1 = self.frames[self._p1[i]]  # F_{t-1}
        f0 = self.frames[i]  # current (F_t)
        stack = np.concatenate([f2, f1, f0], axis=2)  # (H, W, 9)
        stack = np.ascontiguousarray(stack.transpose(2, 0, 1))  # (9, H, W)
        return {
            "frames": torch.from_numpy(stack),
            "prev_actions": torch.tensor([self.prev0[i], self.prev1[i]], dtype=torch.long),
            "turn_label": torch.tensor(int(self.turn[i]), dtype=torch.long),
            "shoot_label": torch.tensor(int(self.shoot[i]), dtype=torch.long),
        }


# =============================================================================
# Eval + train
# =============================================================================


def evaluate(model, loader, device, bf16):
    model.eval()
    ct = cs = call = total = 0
    dtype = torch.bfloat16 if bf16 else torch.float32
    with torch.no_grad():
        for b in loader:
            frames = b["frames"].to(device)
            prev = b["prev_actions"].to(device)
            tl = b["turn_label"].to(device)
            sl = b["shoot_label"].to(device)
            with torch.amp.autocast(device.type, dtype=dtype, enabled=bf16):
                r = model(frames, prev)
            pt = r["turn_logits"].argmax(-1)
            ps = r["shoot_logits"].argmax(-1)
            okt = pt == tl
            oks = ps == sl
            ct += okt.sum().item()
            cs += oks.sum().item()
            call += (okt & oks).sum().item()
            total += tl.size(0)
    acc = {"turn": ct / total, "shoot": cs / total, "all": call / total}
    print(f"\n  Eval: {total} samples | turn={acc['turn']:.3f} shoot={acc['shoot']:.3f} all={acc['all']:.3f}")
    model.train()
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data",
        default="data/cnn-oracle",
        help="Local dir with frames.u8 + labels.npz, OR a HuggingFace Hub dataset repo id (snapshot_downloaded).",
    )
    ap.add_argument(
        "--base-model",
        default="models/doom-cnn-4L-no-fwd",
        help="Encoder dir from create_model.py (config + weights).",
    )
    ap.add_argument("--output", default="output/cnn-sft")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--constant-lr", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Early stop after this many consecutive evals with no improvement in val "
        "all-accuracy (0 = disabled). Counted in EVAL units, not epochs, so it adapts to "
        "dataset size (~4 epochs on a 100K set at the default eval-steps).",
    )
    ap.add_argument(
        "--min-delta", type=float, default=0.0, help="Minimum val all-accuracy gain to count as an improvement."
    )
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 60)
    print("DOOM Conv-Stem — SFT (behavior cloning)")
    print("=" * 60)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    print(f"\nLoading data from {args.data}...")
    full = ConvStemFrameDataset(args.data)
    print(f"  {len(full)} samples")
    dist = full.label_distribution()
    print(f"  turn L/N/R = {dist['turn_L']:.3f}/{dist['turn_N']:.3f}/{dist['turn_R']:.3f}  shoot={dist['shoot']:.3f}")

    eval_size = max(1, len(full) // 10)
    train_size = len(full) - eval_size
    train_ds, eval_ds = torch.utils.data.random_split(
        full, [train_size, eval_size], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  Train: {train_size}, Eval: {eval_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    print(f"\nBuilding model from {args.base_model}...")
    model = DoomConvStemClassifier(args.base_model).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_stem = sum(p.numel() for p in model.stem.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"  stem {n_stem:,} | encoder {n_enc:,} | total {n_total:,}")

    # Head biases init to log(class priors): each head starts at its prior, so the
    # only loss-decreasing direction is input-conditional features. This removes the
    # cold-start "always predict the majority class" attractor.
    with torch.no_grad():
        p_turn = torch.tensor(
            [dist["turn_L"], dist["turn_N"], dist["turn_R"]], dtype=torch.float32, device=device
        ).clamp(min=1e-6)
        p_shoot = torch.tensor([1 - dist["shoot"], dist["shoot"]], dtype=torch.float32, device=device).clamp(min=1e-6)
        model.turn_head.bias.copy_(torch.log(p_turn))
        model.shoot_head.bias.copy_(torch.log(p_shoot))
    print("  Initialized head biases to log(class priors).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs

    def get_lr(step):
        if args.constant_lr:
            return 1.0
        if step < args.warmup_steps:
            return max(0.01, (step + 1) / max(1, args.warmup_steps))
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    os.makedirs(args.output, exist_ok=True)

    def save_ckpt(path):
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        for fn in ("config.json", "stem_config.json"):
            src = os.path.join(args.base_model, fn)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(path, fn))

    es_desc = f"patience={args.patience} evals" if args.patience else "off"
    print(
        f"\n  epochs={args.epochs} (max) bs={args.batch_size} lr={args.lr} "
        f"steps={total_steps} bf16={args.bf16} early-stop={es_desc}\nStarting training..."
    )
    step = 0
    best = 0.0
    no_improve = 0
    stop = False
    rl = rt = rs = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        if stop:
            break
        model.train()
        for b in train_loader:
            frames = b["frames"].to(device)
            prev = b["prev_actions"].to(device)
            kw = {"turn_labels": b["turn_label"].to(device), "shoot_labels": b["shoot_label"].to(device)}
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, dtype=dtype, enabled=args.bf16):
                r = model(frames, prev, **kw)
                loss = r["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            rl += loss.item()
            rt += r["loss_turn"].item()
            rs += r["loss_shoot"].item()
            step += 1
            if step % args.logging_steps == 0:
                k = args.logging_steps
                print(
                    f"  step {step:5d} | ep {epoch + 1} | loss {rl / k:.4f} "
                    f"(t={rt / k:.4f} s={rs / k:.4f}) | "
                    f"lr {scheduler.get_last_lr()[0]:.6f} | {time.time() - t0:.0f}s"
                )
                rl = rt = rs = 0.0
            if step % args.eval_steps == 0:
                acc = evaluate(model, eval_loader, device, args.bf16)
                if acc["all"] > best + args.min_delta:
                    best = acc["all"]
                    no_improve = 0
                    save_ckpt(os.path.join(args.output, "best"))
                    print(f"  New best all={acc['all']:.3f} -> saved.")
                elif args.patience:
                    no_improve += 1
                    print(f"  No improvement ({no_improve}/{args.patience}); best={best:.3f}")
                    if no_improve >= args.patience:
                        print(
                            f"  Early stopping at step {step} (epoch {epoch + 1}): "
                            f"val all flat for {args.patience} evals."
                        )
                        stop = True
                        break

    print("\n" + "=" * 60 + "\nFinal evaluation:")
    final = evaluate(model, eval_loader, device, args.bf16)
    print(f"\nBest all : {best:.3f}\nFinal all: {final['all']:.3f}")
    save_ckpt(os.path.join(args.output, "final"))
    print(f"Saved to {args.output}/final")


if __name__ == "__main__":
    main()
