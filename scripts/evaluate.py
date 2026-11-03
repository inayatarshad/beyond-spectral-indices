"""Test-set evaluation and the stratified "when" analysis: python scripts/evaluate.py configs/burn.yaml

Writes to runs/<task>/eval/:
  overall.csv    pooled metrics per method, with 95% bootstrap CIs over chips
  strata.csv     metrics and error counts per (family, stratum, method)
  gain.csv       share of the error reduction vs index_tuned that falls in each stratum
  per_chip.csv   chip-level counts, used to pick qualitative examples
"""

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bsi.config import load_config
from bsi.data.chips import IGNORE, chip_paths, compute_band_stats
from bsi.metrics import Confusion, summarize
from bsi.predict import ALL_METHODS, Predictor, load_thresholds
from bsi.strata import chip_strata

REFERENCE = "index_tuned"
N_BOOT = 1000

parser = argparse.ArgumentParser()
parser.add_argument("config")
cfg = load_config(parser.parse_args().config)
out = cfg["run_dir"] / "eval"
out.mkdir(exist_ok=True)


def available(m):
    if m.startswith("index"):
        return True
    return (cfg["run_dir"] / m / ("model.pt" if m.startswith("unet") else "model.txt")).exists()


methods = [m for m in ALL_METHODS if available(m)]
print("methods:", methods)
device = "cuda" if torch.cuda.is_available() else "cpu"
predict = Predictor(cfg, methods, compute_band_stats(cfg["data_dir"]), device)
tuned = load_thresholds(cfg)["tuned"]

strata_conf = {}  # (family, method) -> Confusion
chip_rows = []
for path in tqdm(chip_paths(cfg["data_dir"], "test")):
    chip, preds = predict.chip(path)
    valid = chip["mask"] != IGNORE
    target = chip["mask"] == 1
    strata = chip_strata(chip, cfg, tuned)
    for m, pred in preds.items():
        tp, fp = np.sum(pred & target & valid), np.sum(pred & ~target & valid)
        fn, tn = np.sum(~pred & target & valid), np.sum(~pred & ~target & valid)
        chip_rows.append({"chip": path.stem, "method": m, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        for family, labels in strata.items():
            strata_conf.setdefault((family, m), Confusion()).update(pred, target, valid, labels)

per_chip = pd.DataFrame(chip_rows)
per_chip.to_csv(out / "per_chip.csv", index=False)

# pooled metrics with a chip-level bootstrap
rng = np.random.default_rng(cfg["seed"])
chips = per_chip["chip"].unique()
boot_idx = rng.integers(0, len(chips), size=(N_BOOT, len(chips)))
rows = []
ref_boot = None
for m in methods:
    c = per_chip[per_chip.method == m].set_index("chip").loc[chips, ["tp", "fp", "fn", "tn"]].to_numpy()
    row = {"method": m, **summarize(*c.sum(0))}
    s = c[boot_idx].sum(1)
    boot_iou = s[:, 0] / np.maximum(s[:, 0] + s[:, 1] + s[:, 2], 1)
    row["iou_lo"], row["iou_hi"] = np.percentile(boot_iou, [2.5, 97.5])
    if m == REFERENCE:
        ref_boot = boot_iou
    rows.append((row, boot_iou))
overall = []
for row, boot_iou in rows:
    row["p_better_than_ref"] = float(np.mean(boot_iou > ref_boot)) if ref_boot is not None else np.nan
    overall.append(row)
overall = pd.DataFrame(overall)
overall.to_csv(out / "overall.csv", index=False)
print(overall[["method", "iou", "iou_lo", "iou_hi", "f1", "precision", "recall"]].round(4).to_string(index=False))

# stratified table
srows = []
for (family, m), conf in strata_conf.items():
    for stratum, counts in conf.counts.items():
        tp, fp, fn, tn = counts
        srows.append({"family": family, "stratum": stratum, "method": m, "errors": fp + fn,
                      "fp": fp, "fn": fn, **summarize(tp, fp, fn, tn)})
strata_df = pd.DataFrame(srows)
strata_df.to_csv(out / "strata.csv", index=False)

# where does each method's error reduction over the reference come from?
grows = []
for family, fam in strata_df.groupby("family"):
    ref = fam[fam.method == REFERENCE].set_index("stratum")
    for m, g in fam.groupby("method"):
        if m == REFERENCE:
            continue
        g = g.set_index("stratum")
        reduction = ref["errors"] - g["errors"]
        total = reduction.sum()
        for stratum in g.index:
            grows.append({"family": family, "stratum": stratum, "method": m,
                          "pixel_share": g.loc[stratum, "pixels"] / g["pixels"].sum(),
                          "error_reduction": int(reduction[stratum]),
                          "reduction_share": reduction[stratum] / total if total else np.nan})
pd.DataFrame(grows).to_csv(out / "gain.csv", index=False)
print(f"wrote {out}")
