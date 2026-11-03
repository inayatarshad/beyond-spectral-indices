"""The method ladder.

Learned methods form a 2x2 design, {index bands, all bands} x {per-pixel, spatial}:

    lgbm_index   LightGBM per pixel, only the bands the index uses
    lgbm_all     LightGBM per pixel, all bands
    unet_index   U-Net, only the bands the index uses
    unet_all     U-Net, all bands

Comparing cells separates the gain from a learned non-linear decision rule,
from extra spectral bands, and from spatial context.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

from .bands import channel_names, index_channel_subset
from .data.chips import IGNORE, chip_paths, load_chip

LEARNED = ["lgbm_index", "lgbm_all", "unet_index", "unet_all"]


def channels_for(variant: str, task: str, use_pre: bool) -> list[int]:
    if variant.endswith("_index"):
        return index_channel_subset(task, use_pre)
    return list(range(len(channel_names(task, use_pre))))


class ChannelSelectUNet(nn.Module):
    """U-Net that reads a fixed subset of the input channels.

    Taking the full stack as input keeps attribution maps comparable across
    variants: unused channels simply receive zero attribution.
    """

    def __init__(self, channels: list[int], encoder: str, encoder_weights: str | None):
        super().__init__()
        self.register_buffer("channels", torch.tensor(channels, dtype=torch.long))
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                            in_channels=len(channels), classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.index_select(1, self.channels)).squeeze(1)  # logits (B, H, W)


def build_unet(cfg: dict, variant: str, weights: bool = True) -> ChannelSelectUNet:
    u = cfg["unet"]
    return ChannelSelectUNet(channels_for(variant, cfg["task"], cfg["use_pre"]), u["encoder"],
                             u["encoder_weights"] if weights else None)


def load_unet(cfg: dict, variant: str, device: str = "cpu") -> ChannelSelectUNet:
    model = build_unet(cfg, variant, weights=False)
    model.load_state_dict(torch.load(Path(cfg["run_dir"]) / variant / "model.pt", map_location=device))
    return model.to(device).eval()


def sample_pixels(data_dir: Path, split: str, channels: list[int], per_chip: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for p in chip_paths(data_dir, split):
        chip = load_chip(p)
        m = chip["mask"].ravel()
        valid = np.flatnonzero(m != IGNORE)
        if valid.size == 0:
            continue
        pick = rng.choice(valid, min(per_chip, valid.size), replace=False)
        img = chip["image"][channels].reshape(len(channels), -1)
        xs.append(img[:, pick].T)
        ys.append(m[pick])
    return np.concatenate(xs), np.concatenate(ys)


def train_lgbm(cfg: dict, variant: str) -> lgb.LGBMClassifier:
    p = cfg["lgbm"]
    channels = channels_for(variant, cfg["task"], cfg["use_pre"])
    x_tr, y_tr = sample_pixels(cfg["data_dir"], "train", channels, p["pixels_per_chip"], cfg["seed"])
    x_va, y_va = sample_pixels(cfg["data_dir"], "val", channels, p["pixels_per_chip"], cfg["seed"] + 1)
    model = lgb.LGBMClassifier(num_leaves=p["num_leaves"], n_estimators=p["n_estimators"],
                               learning_rate=p["learning_rate"], random_state=cfg["seed"], verbose=-1)
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
    out = Path(cfg["run_dir"]) / variant
    out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))
    return model


def load_lgbm(cfg: dict, variant: str) -> lgb.Booster:
    return lgb.Booster(model_file=str(Path(cfg["run_dir"]) / variant / "model.txt"))


def lgbm_predict_chip(booster: lgb.Booster, image: np.ndarray, channels: list[int]) -> np.ndarray:
    c, h, w = image.shape
    feats = image[channels].reshape(len(channels), -1).T
    return booster.predict(feats).reshape(h, w)
