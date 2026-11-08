"""The "why" analysis.

  band_attribution   Integrated Gradients and GradientSHAP on U-Net outputs, per channel
  context_profile    how far from a pixel the evidence for that pixel lies (U-Net)
  occlusion          drop in IoU when a band is replaced by its train mean (any method)
  tree_shap          exact TreeSHAP for the LightGBM rungs
  randomization      sanity check: attributions of a re-initialised network should differ
"""

import numpy as np
import shap
import torch
from captum.attr import GradientShap, IntegratedGradients
from scipy.stats import spearmanr

from .data.chips import IGNORE


class RegionLogit(torch.nn.Module):
    """Scalar output for attribution: mean logit over a fixed pixel region."""

    def __init__(self, model, region: torch.Tensor):
        super().__init__()
        self.model, self.region = model, region.float()

    def forward(self, x):
        logits = self.model(x)
        return (logits * self.region).sum((1, 2)) / self.region.sum().clamp(min=1)


def band_attribution(model, x: torch.Tensor, region: torch.Tensor, steps: int,
                     baselines: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """Per-channel share of |attribution| inside the image for one normalised chip x (1, C, H, W).

    The IG baseline is the all-zero tensor, i.e. the train-mean spectrum after normalisation.
    """
    f = RegionLogit(model, region)
    ig = IntegratedGradients(f).attribute(x, baselines=torch.zeros_like(x), n_steps=steps, internal_batch_size=8)
    out = {"ig": ig[0].abs().sum((1, 2)).detach().cpu().numpy()}
    if baselines is not None:
        gs = GradientShap(f).attribute(x, baselines=baselines, n_samples=steps, stdevs=0.0)
        out["gradshap"] = gs[0].abs().sum((1, 2)).detach().cpu().numpy()
    return {k: v / (v.sum() + 1e-12) for k, v in out.items()}


class PixelLogit(torch.nn.Module):
    def __init__(self, model, r: int, c: int):
        super().__init__()
        self.model, self.r, self.c = model, r, c

    def forward(self, x):
        return self.model(x)[:, self.r, self.c]


def context_profile(model, x: torch.Tensor, pixels: list[tuple[int, int]], steps: int) -> np.ndarray:
    """Cumulative share of |IG| mass within radius d of each target pixel, d = 0..chip diagonal (pixels)."""
    _, _, h, w = x.shape
    yy, xx = np.mgrid[0:h, 0:w]
    n_bins = int(np.ceil(np.hypot(h, w))) + 1
    curves = []
    for r, c in pixels:
        attr = IntegratedGradients(PixelLogit(model, r, c)).attribute(
            x, baselines=torch.zeros_like(x), n_steps=steps, internal_batch_size=8)
        mass = attr[0].abs().sum(0).detach().cpu().numpy()
        dist = np.rint(np.hypot(yy - r, xx - c)).astype(int)
        hist = np.bincount(dist.ravel(), mass.ravel(), minlength=n_bins)[:n_bins]
        curves.append(np.cumsum(hist) / (mass.sum() + 1e-12))
    return np.array(curves)


def radius_at(curves: np.ndarray, share: float) -> np.ndarray:
    """Smallest radius whose disc holds `share` of the attribution mass, per curve."""
    return np.argmax(curves >= share - 1e-9, axis=1)


def occlusion(predict_fn, chips: list[dict], groups: dict[str, list[int]], mean: np.ndarray) -> dict[str, float]:
    """IoU drop per channel group when those channels are set to their train mean.

    predict_fn(image) -> bool map. Returns {group: iou_full - iou_occluded}, pooled over chips.
    """

    def pooled_iou(transform):
        tp = fp = fn = 0
        for chip in chips:
            pred = predict_fn(transform(chip["image"]))
            valid = chip["mask"] != IGNORE
            t = chip["mask"] == 1
            tp += np.sum(pred & t & valid)
            fp += np.sum(pred & ~t & valid)
            fn += np.sum(~pred & t & valid)
        return tp / max(tp + fp + fn, 1)

    base = pooled_iou(lambda im: im)
    drops = {}
    for name, chans in groups.items():
        def occlude(im, chans=chans):
            im = im.copy()
            im[chans] = mean[chans, None, None]
            return im
        drops[name] = base - pooled_iou(occlude)
    return {"_full_iou": base, **drops}


def tree_shap(booster, x: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature for a LightGBM booster on pixel features x (N, F)."""
    sv = shap.TreeExplainer(booster).shap_values(x)
    sv = sv[1] if isinstance(sv, list) else sv
    return np.abs(sv).mean(0)


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).statistic)
