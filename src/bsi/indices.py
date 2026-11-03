"""Spectral-index baselines and threshold selection.

Three thresholds per index, from weakest to strongest assumption:
  literature  fixed value from the remote-sensing literature (no labels used)
  otsu        per-chip Otsu threshold (no labels used)
  tuned       single global threshold maximising IoU on the train split (uses labels)
"tuned" is the fair comparison against learned models, which also see train labels.
"""

import numpy as np
from skimage.filters import threshold_otsu

from .bands import channel_index, channel_names

EPS = 1e-6


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def compute_index(image: np.ndarray, task: str, name: str, use_pre: bool = True) -> np.ndarray:
    """Index value per pixel, oriented so that larger means "target" (burned / tree)."""
    names = channel_names(task, use_pre)
    if name == "ndvi":
        return _nd(image[channel_index(names, "B08")], image[channel_index(names, "B04")])
    if name in ("nbr", "dnbr"):
        p = "post_" if task == "burn" else ""
        nbr_post = _nd(image[channel_index(names, "B08", p)], image[channel_index(names, "B12", p)])
        if name == "nbr":
            return -nbr_post  # burned surfaces have low NBR
        nbr_pre = _nd(image[channel_index(names, "B08", "pre_")], image[channel_index(names, "B12", "pre_")])
        return nbr_pre - nbr_post
    raise ValueError(f"unknown index {name!r}")


def otsu_threshold(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size < 2 or v.min() == v.max():
        return float("inf")
    return float(threshold_otsu(v))


def tune_threshold(index_values: list[np.ndarray], masks: list[np.ndarray], grid: np.ndarray | None = None,
                   max_pixels_per_chip: int = 20000, seed: int = 0) -> float:
    """Global threshold with the best pooled IoU over the given chips (pixels subsampled per chip)."""
    grid = np.linspace(-1, 1, 401) if grid is None else grid
    rng = np.random.default_rng(seed)
    vs, ys = [], []
    for iv, m in zip(index_values, masks):
        keep = np.flatnonzero((m != 255).ravel() & np.isfinite(iv).ravel())
        if keep.size > max_pixels_per_chip:
            keep = rng.choice(keep, max_pixels_per_chip, replace=False)
        vs.append(iv.ravel()[keep])
        ys.append(m.ravel()[keep])
    v = np.concatenate(vs)
    y = np.concatenate(ys).astype(bool)
    order = np.argsort(v)
    v, y = v[order], y[order]
    # cumulative positives from the top let every threshold be scored in O(n log n)
    pos_above = np.cumsum(y[::-1])[::-1]
    n_pos = y.sum()
    best_t, best_iou = grid[0], -1.0
    for t in grid:
        k = np.searchsorted(v, t, side="right")
        tp = pos_above[k] if k < len(v) else 0
        pred = len(v) - k
        iou = tp / (pred + n_pos - tp + EPS)
        if iou > best_iou:
            best_t, best_iou = t, iou
    return float(best_t)
