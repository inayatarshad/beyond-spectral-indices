"""Common on-disk chip format shared by both tasks.

Each chip is data/<task>/chips/<split>/<chip_id>.npz with
    image  float32 (C, H, W)  surface reflectance, roughly 0..1
    mask   uint8   (H, W)     1 = target, 0 = background, 255 = no label
and optionally
    worldcover uint8 (H, W)   ESA WorldCover class codes (forest task strata)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE = 255
SPLITS = ("train", "val", "test")


def save_chip(data_dir: Path, split: str, chip_id: str, image: np.ndarray, mask: np.ndarray, **aux) -> None:
    out = Path(data_dir) / "chips" / split
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{chip_id}.npz", image=image.astype(np.float32), mask=mask.astype(np.uint8), **aux)


def chip_paths(data_dir: Path, split: str) -> list[Path]:
    return sorted((Path(data_dir) / "chips" / split).glob("*.npz"))


def load_chip(path: Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def compute_band_stats(data_dir: Path) -> dict:
    """Per-channel mean/std over the train split, cached next to the chips."""
    cache = Path(data_dir) / "band_stats.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {"mean": z["mean"], "std": z["std"]}
    total, total_sq, count = None, None, 0
    for p in chip_paths(data_dir, "train"):
        img = load_chip(p)["image"].astype(np.float64)
        flat = img.reshape(img.shape[0], -1)
        total = flat.sum(1) if total is None else total + flat.sum(1)
        total_sq = (flat**2).sum(1) if total_sq is None else total_sq + (flat**2).sum(1)
        count += flat.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
    np.savez(cache, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class ChipDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, stats: dict, augment: bool = False, crop: int | None = None):
        self.paths = chip_paths(data_dir, split)
        if not self.paths:
            raise FileNotFoundError(f"no chips in {data_dir}/chips/{split}; run scripts/prepare_data.py first")
        self.mean = stats["mean"][:, None, None]
        self.std = stats["std"][:, None, None]
        self.augment = augment
        self.crop = crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        chip = load_chip(self.paths[i])
        x = (chip["image"] - self.mean) / self.std
        y = chip["mask"].astype(np.int64)
        if self.crop and self.crop < y.shape[0]:
            r, c = np.random.randint(0, y.shape[0] - self.crop + 1, size=2)
            x, y = x[:, r : r + self.crop, c : c + self.crop], y[r : r + self.crop, c : c + self.crop]
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
