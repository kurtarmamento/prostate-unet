# src/prostate_unet/datasets.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import csv, random
import numpy as np
import torch
from torch.utils.data import Dataset

class Slices2D(Dataset):
    """
    Samples axial 2D slices from cached cropped volumes (.npz).
    - Always returns fixed-size tensors: (C=k, H=crop_size, W=crop_size)
    - Train: random crop if larger; center-pad if smaller
    - Val/Test: center-crop if larger; center-pad if smaller
    - Optional 2.5D context with k neighboring slices as channels
    - Simple flips for augmentation (train only)
    """
    def __init__(
        self,
        manifest_csv: str | Path,   # e.g., artifacts/cache_splits.csv
        split: str,                 # train | val | test
        crop_size: int = 256,       # final H=W
        k_slices: int = 1,          # 1 for 2D; odd (3/5) for 2.5D
        pos_fraction: float = 0.5,  # target fraction of positive slices in train
        seed: int = 1337,
    ):
        assert k_slices % 2 == 1, "k_slices should be odd (1,3,5,...)"
        self.rng = random.Random(seed)
        self.split = split
        self.crop = crop_size
        self.k = k_slices
        rows = list(csv.DictReader(Path(manifest_csv).open()))
        self.items = [r for r in rows if r.get("split", "train") == split]
        if not self.items:
            raise ValueError(f"No rows for split={split} in {manifest_csv}")

        # Pre-index which z-slices have foreground per volume
        self._index: List[Dict] = []
        for r in self.items:
            d = np.load(r["cache_path"])
            m = d["label"].astype(np.uint8)  # (Z, Y, X)
            pos = np.where(m.sum(axis=(1, 2)) > 0)[0].tolist()
            allz = list(range(m.shape[0]))
            self._index.append({
                "path": r["cache_path"],
                "Z": m.shape[0], "Y": m.shape[1], "X": m.shape[2],
                "pos": pos,
                "neg": [z for z in allz if z not in set(pos)],
            })

        # define epoch length (heuristic)
        self.samples_per_epoch = max(4, 2 * len(self._index))
        self.pos_fraction = pos_fraction

    def __len__(self) -> int:
        return self.samples_per_epoch if self.split == "train" else len(self._index)

    # ---- helpers: crop/pad -------------------------------------------------

    def _center_crop_hw(self, a: np.ndarray, H: int, W: int) -> np.ndarray:
        """2D array (H,W) center-crop (or return as-is if smaller)."""
        h, w = a.shape
        if h <= H or w <= W:
            return a
        y0 = (h - H) // 2
        x0 = (w - W) // 2
        return a[y0:y0+H, x0:x0+W]

    def _random_crop_hw(self, a: np.ndarray, H: int, W: int, rng: random.Random) -> np.ndarray:
        """2D array (H,W) random crop; assumes a is larger than target."""
        h, w = a.shape
        if h <= H or w <= W:
            return self._center_crop_hw(a, H, W)
        y0 = rng.randint(0, h - H)
        x0 = rng.randint(0, w - W)
        return a[y0:y0+H, x0:x0+W]

    def _pad_to_hw(self, a: np.ndarray, H: int, W: int, constant: int = 0) -> np.ndarray:
        """Center-pad 2D array (H,W) to exactly (H,W)."""
        h, w = a.shape
        ph = max(0, H - h); pw = max(0, W - w)
        top = ph // 2; bottom = ph - top
        left = pw // 2; right = pw - left
        if ph == 0 and pw == 0:
            return a
        return np.pad(a, ((top, bottom), (left, right)), mode="constant", constant_values=constant)

    def _pad_to_chw_edge(self, a: np.ndarray, H: int, W: int) -> np.ndarray:
        """Center-pad 3D array (C,H,W) using 'edge' (replicate) for image channels."""
        C, h, w = a.shape
        ph = max(0, H - h); pw = max(0, W - w)
        top = ph // 2; bottom = ph - top
        left = pw // 2; right = pw - left
        if ph == 0 and pw == 0:
            return a
        return np.pad(a, ((0, 0), (top, bottom), (left, right)), mode="edge")

    # -----------------------------------------------------------------------

    def __getitem__(self, idx: int):
        # pick volume
        if self.split == "train":
            meta = self.rng.choice(self._index)
        else:
            meta = self._index[idx]

        d = np.load(meta["path"])
        vol = d["image"].astype(np.float32)  # (Z, Y, X) z-scored already
        msk = d["label"].astype(np.uint8)    # (Z, Y, X)

        # choose slice index
        if self.split == "train":
            want_pos = (self.rng.random() < self.pos_fraction) and len(meta["pos"]) > 0
            z = self.rng.choice(meta["pos"] if want_pos else (meta["neg"] or meta["pos"] or [0]))
        else:
            z = vol.shape[0] // 2  # deterministic preview; val metrics scan full Z elsewhere

        # build 2D arrays
        if self.k == 1:
            img2d = vol[z:z+1]       # (1, Y, X)
        else:
            half = self.k // 2
            zs = [min(max(zz, 0), vol.shape[0] - 1) for zz in range(z - half, z + half + 1)]
            img2d = vol[zs]          # (k, Y, X)
        msk2d = msk[z]                # (Y, X)

        # crop → pad to fixed (crop_size, crop_size)
        H = W = self.crop
        # crop (random for train; center for val/test) if larger
        if img2d.shape[1] > H or img2d.shape[2] > W:
            if self.split == "train":
                # apply same random crop to all channels and mask
                # we compute on mask shape to get y0/x0, then use them for image
                h, w = msk2d.shape
                y0 = self.rng.randint(0, max(0, h - H)) if h > H else 0
                x0 = self.rng.randint(0, max(0, w - W)) if w > W else 0
                img2d = img2d[:, y0:y0+H, x0:x0+W]
                msk2d = msk2d[y0:y0+H, x0:x0+W]
            else:
                # center-crop
                img2d = img2d[:, max(0,(img2d.shape[1]-H)//2):max(0,(img2d.shape[1]-H)//2)+H,
                                   max(0,(img2d.shape[2]-W)//2):max(0,(img2d.shape[2]-W)//2)+W]
                msk2d = self._center_crop_hw(msk2d, H, W)

        # pad (if smaller) — images with edge replicate, mask with zeros
        if img2d.shape[1] < H or img2d.shape[2] < W:
            img2d = self._pad_to_chw_edge(img2d, H, W)
        if msk2d.shape[0] < H or msk2d.shape[1] < W:
            msk2d = self._pad_to_hw(msk2d, H, W, constant=0)

        # simple flips (train only)
        if self.split == "train":
            if self.rng.random() < 0.5:
                img2d = np.flip(img2d, axis=2); msk2d = np.flip(msk2d, axis=1)
            if self.rng.random() < 0.5:
                img2d = np.flip(img2d, axis=1); msk2d = np.flip(msk2d, axis=0)

        # to tensors (copy -> contiguous)
        x = torch.from_numpy(img2d.copy())            # (C, H, W) float32
        y = torch.from_numpy(msk2d[None, ...].copy()) # (1, H, W) uint8→float later
        return x, y, {"z": z, "path": meta["path"]}
