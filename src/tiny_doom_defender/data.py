"""The recorded-oracle dataset: frame stacking over frames.u8 + labels.npz."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from tiny_doom_defender.constants import RES_H, RES_W
from tiny_doom_defender.utils import stack_channels


class ConvStemFrameDataset(Dataset):
    """Memory-maps `frames.u8` and, per index, builds the 9-channel stack
    [F_{t-2}, F_{t-1}, F_t] with episode-boundary clamping (at episode start the
    current frame is repeated, so no motion leaks across episodes). prev0/prev1 come
    from the label columns. Frames stay uint8; the stem casts + normalizes on device.
    """

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
                f"constants.py geometry than the current one."
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
        stack = stack_channels([f2, f1, f0])  # (9, H, W)
        return {
            "frames": torch.from_numpy(stack),
            "prev_actions": torch.tensor([self.prev0[i], self.prev1[i]], dtype=torch.long),
            "turn_label": torch.tensor(int(self.turn[i]), dtype=torch.long),
            "shoot_label": torch.tensor(int(self.shoot[i]), dtype=torch.long),
        }
