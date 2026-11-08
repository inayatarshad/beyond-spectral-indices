"""Aggregate cross-validation runs: python scripts/aggregate_cv.py runs/burn_cv

Writes to <runs_root>/cv/:
  per_fold.csv      metrics per (test fold, seed, method)
  summary.csv       per method: mean/std/min/max IoU over folds (seed 0), seed std on the canonical fold,
                    and pooled out-of-fold IoU with a 95 % chip-bootstrap CI
  steps.csv         each step of the factorial ladder: mean/std over folds, folds improved,
                    and the pooled paired-bootstrap CI with P(step > 0)

Under 5-fold CV every chip is tested exactly once, so the "pooled" columns cover all of CaBuAr.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = {  # name: (from, to)
    "threshold": ("index_literature", "index_tuned"),
    "rule": ("index_tuned", "lgbm_index"),
    "spectrum_pixel": ("lgbm_index", "lgbm_all"),
    "space_index_bands": ("lgbm_index", "unet_index"),
    "space_all_bands": ("lgbm_all", "unet_all"),
    "spectrum_spatial": ("unet_index", "unet_all"),
    "total_best": ("index_tuned", "unet_index"),
}
N_BOOT = 2000

parser = argparse.ArgumentParser()
parser.add_argument("runs_root")
parser.add_argument("--canonical-fold", type=int, default=4)
args = parser.parse_args()
root = Path(args.runs_root)
out = root / "cv"
out.mkdir(exist_ok=True)

rows, chips = [], []
for d in sorted(root.glob("test*_seed*")):
    m = re.fullmatch(r"test(\d+)_seed(\d+)", d.name)
    if not m or not (d / "eval" / "overall.csv").exists():
        continue
    test, seed = int(m.group(1)), int(m.group(2))
    o = pd.read_csv(d / "eval" / "overall.csv")
    rows.append(o.assign(test=test, seed=seed))
    if seed == 0:
        chips.append(pd.read_csv(d / "eval" / "per_chip.csv").assign(test=test))
if not rows:
    raise SystemExit(f"no finished runs under {root}")
per_fold = pd.concat(rows, ignore_index=True)
per_fold[["test", "seed", "method", "iou", "f1", "precision", "recall"]].to_csv(out / "per_fold.csv", index=False)

s0 = per_fold[per_fold.seed == 0]
summary = s0.groupby("method").agg(n_folds=("iou", "size"), iou_mean=("iou", "mean"), iou_std=("iou", "std"),
                                   iou_min=("iou", "min"), iou_max=("iou", "max"), f1_mean=("f1", "mean"),
                                   precision_mean=("precision", "mean"), recall_mean=("recall", "mean"))
canon = per_fold[per_fold.test == args.canonical_fold]
summary["seed_std_canonical"] = canon.groupby("method").iou.std()
summary["n_seeds_canonical"] = canon.groupby("method").iou.size()

# pooled out-of-fold IoU with chip bootstrap
pc = pd.concat(chips, ignore_index=True)
pc["key"] = pc.test.astype(str) + "/" + pc.chip
tab = {k: g.set_index("key")[["tp", "fp", "fn"]] for k, g in pc.groupby("method")}
keys = sorted(set.intersection(*(set(t.index) for t in tab.values())))
arr = {k: t.loc[keys].to_numpy(float) for k, t in tab.items()}  # (n_chips, 3)
rng = np.random.default_rng(0)
boot = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))


def iou(a, idx=None):
    s = a.sum(0) if idx is None else a[idx].sum(1)
    return s[..., 0] / np.maximum(s[..., 0] + s[..., 1] + s[..., 2], 1)


pooled = {k: (iou(a), *np.percentile(iou(a, boot), [2.5, 97.5])) for k, a in arr.items()}
summary["iou_pooled"] = pd.Series({k: v[0] for k, v in pooled.items()})
summary["iou_pooled_lo"] = pd.Series({k: v[1] for k, v in pooled.items()})
summary["iou_pooled_hi"] = pd.Series({k: v[2] for k, v in pooled.items()})
summary["n_chips_pooled"] = len(keys)
summary.to_csv(out / "summary.csv")

steps = []
wide = s0.pivot(index="test", columns="method", values="iou")
for name, (a, b) in STEPS.items():
    if a not in wide or b not in wide or a not in arr or b not in arr:
        continue
    d = wide[b] - wide[a]
    db = iou(arr[b], boot) - iou(arr[a], boot)
    steps.append({"step": name, "from": a, "to": b, "fold_mean": d.mean(), "fold_std": d.std(),
                  "folds_improved": int((d > 0).sum()), "n_folds": int(d.notna().sum()),
                  "pooled": iou(arr[b]) - iou(arr[a]), "pooled_lo": np.percentile(db, 2.5),
                  "pooled_hi": np.percentile(db, 97.5), "p_positive": float((db > 0).mean())})
steps = pd.DataFrame(steps)
steps.to_csv(out / "steps.csv", index=False)

pd.set_option("display.width", 200, "display.precision", 3)
print(summary[["n_folds", "iou_mean", "iou_std", "seed_std_canonical", "iou_pooled", "iou_pooled_lo", "iou_pooled_hi"]])
print(steps)
