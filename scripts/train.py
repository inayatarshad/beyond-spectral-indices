"""Train one learned rung: python scripts/train.py configs/burn.yaml unet_all

Variants: lgbm_index, lgbm_all, unet_index, unet_all.
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bsi.config import load_config
from bsi.data.chips import IGNORE, ChipDataset, compute_band_stats
from bsi.models import LEARNED, build_unet, train_lgbm

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("variant", choices=LEARNED)
args = parser.parse_args()
cfg = load_config(args.config)
out = cfg["run_dir"] / args.variant
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])

if args.variant.startswith("lgbm"):
    t0 = time.time()
    model = train_lgbm(cfg, args.variant)
    (out / "train_log.json").write_text(json.dumps({"best_iteration": model.best_iteration_,
                                                    "seconds": time.time() - t0}, indent=2))
    raise SystemExit


def loss_fn(logits, y):
    valid = (y != IGNORE).float()
    t = (y == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * valid).sum() + 1)
    return bce + dice


def val_iou(model, loader, device):
    tp = fp = fn = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x) > 0
            valid, t = y != IGNORE, y == 1
            tp += (pred & t & valid).sum().item()
            fp += (pred & ~t & valid).sum().item()
            fn += (~pred & t & valid).sum().item()
    return tp / max(tp + fp + fn, 1)


device = "cuda" if torch.cuda.is_available() else "cpu"
stats = compute_band_stats(cfg["data_dir"])
u = cfg["unet"]
train_dl = DataLoader(ChipDataset(cfg["data_dir"], "train", stats, augment=True, crop=u.get("crop_size")),
                      batch_size=u["batch_size"],
                      shuffle=True, num_workers=u["num_workers"], drop_last=True)
val_dl = DataLoader(ChipDataset(cfg["data_dir"], "val", stats), batch_size=u["batch_size"],
                    num_workers=u["num_workers"])
model = build_unet(cfg, args.variant).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=u["lr"], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=u["epochs"])

log, best = [], -1.0
for epoch in range(u["epochs"]):
    model.train()
    t0, total = time.time(), 0.0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    sched.step()
    iou = val_iou(model, val_dl, device)
    log.append({"epoch": epoch, "loss": total / len(train_dl), "val_iou": iou, "seconds": time.time() - t0})
    print(log[-1])
    if iou > best:
        best = iou
        torch.save(model.state_dict(), out / "model.pt")
(out / "train_log.json").write_text(json.dumps(log, indent=2))
print(f"best val IoU {best:.4f}")
