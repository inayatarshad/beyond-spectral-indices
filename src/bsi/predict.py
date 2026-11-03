"""One interface for every method: chip -> binary prediction map."""

import json
from pathlib import Path

import numpy as np
import torch

from .data.chips import load_chip
from .indices import compute_index, otsu_threshold
from .models import LEARNED, channels_for, lgbm_predict_chip, load_lgbm, load_unet

INDEX_METHODS = ["index_literature", "index_otsu", "index_tuned"]
ALL_METHODS = INDEX_METHODS + LEARNED


def load_thresholds(cfg: dict) -> dict:
    return json.loads((Path(cfg["run_dir"]) / "index_thresholds.json").read_text())


class Predictor:
    def __init__(self, cfg: dict, methods: list[str], stats: dict, device: str = "cpu"):
        self.cfg, self.methods, self.device = cfg, methods, device
        self.mean, self.std = stats["mean"][:, None, None], stats["std"][:, None, None]
        self.thresholds = load_thresholds(cfg) if any(m.startswith("index") for m in methods) else {}
        self.unets = {m: load_unet(cfg, m, device) for m in methods if m.startswith("unet")}
        self.lgbms = {m: load_lgbm(cfg, m) for m in methods if m.startswith("lgbm")}

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        cfg, out = self.cfg, {}
        idx = compute_index(image, cfg["task"], cfg["index"]["name"], cfg["use_pre"])
        for m in self.methods:
            if m == "index_literature":
                out[m] = idx > cfg["index"]["literature_threshold"]
            elif m == "index_otsu":
                out[m] = idx > otsu_threshold(idx)
            elif m == "index_tuned":
                out[m] = idx > self.thresholds["tuned"]
            elif m in self.lgbms:
                ch = channels_for(m, cfg["task"], cfg["use_pre"])
                out[m] = lgbm_predict_chip(self.lgbms[m], image, ch) > 0.5
            elif m in self.unets:
                x = torch.from_numpy((image - self.mean) / self.std).float()[None].to(self.device)
                out[m] = (self.unets[m](x)[0] > 0).cpu().numpy()
        return out

    def chip(self, path: Path):
        chip = load_chip(path)
        return chip, self(chip["image"])
