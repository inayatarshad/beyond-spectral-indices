"""Five-fold cross-validation over the CaBuAr folds, resumable.

    python scripts/run_cv.py configs/burn_gpu.yaml --runs-root /content/drive/MyDrive/bsi_cv

Each of the five folds is the test fold once. The fold before it is validation, and
the other three are training (test 4 / val 3 / train 0-2 is the canonical split of the paper).
Every fold runs with seed 0. The canonical fold is repeated with extra seeds to measure
training variance, and only the canonical run is explained at full budget.

The source file is cut into chips once (data/burn_folds/chips/fold<k>). Each fold's
train/val/test directories are then built from symlinks, so the 4 GB of chips are not
copied five times. A step is skipped when its output already exists, so after a
disconnect you just run the same command again.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

N_FOLDS = 5
VARIANTS = ["lgbm_index", "lgbm_all", "unet_index", "unet_all"]

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--runs-root", default="runs/burn_cv", help="where per-fold runs go (use Google Drive on Colab)")
parser.add_argument("--data-root", default="data", help="where chips go (local disk; fast)")
parser.add_argument("--folds", type=int, nargs="+", default=list(range(N_FOLDS)), help="test folds to run")
parser.add_argument("--canonical-fold", type=int, default=4)
parser.add_argument("--extra-seeds", type=int, nargs="*", default=[1, 2], help="extra seeds on the canonical fold")
parser.add_argument("--no-explain", action="store_true")
args = parser.parse_args()

with open(args.config) as f:
    base = yaml.safe_load(f)
runs_root = Path(args.runs_root).resolve()
data_root = Path(args.data_root).resolve()
runs_root.mkdir(parents=True, exist_ok=True)


def run(*cmd):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True)


def write_cfg(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# 1. Cut the source into one chip directory per fold (once).
folds_dir = data_root / "burn_folds"
if not (folds_dir / "chips" / f"fold{N_FOLDS - 1}").exists():
    cfg = dict(base, data_dir=str(folds_dir), run_dir=str(runs_root / "_prepare"))
    cfg["source"] = dict(base["source"], folds={f"fold{k}": [k] for k in range(N_FOLDS)})
    run("scripts/prepare_data.py", write_cfg(runs_root / "_prepare" / "config.yaml", cfg))


def fold_data_dir(test: int) -> Path:
    """data/burn_cv/test<k>/chips/{train,val,test} with symlinks to the per-fold chips."""
    val = (test - 1) % N_FOLDS
    split_of = {k: "test" if k == test else "val" if k == val else "train" for k in range(N_FOLDS)}
    d = data_root / "burn_cv" / f"test{test}"
    if not (d / "chips" / "test").exists():
        for k, split in split_of.items():
            out = d / "chips" / split
            out.mkdir(parents=True, exist_ok=True)
            for p in (folds_dir / "chips" / f"fold{k}").glob("*.npz"):
                link = out / p.name
                if not link.exists():
                    try:
                        os.symlink(p, link)
                    except OSError:  # Windows without symlink rights: hard link (same volume)
                        os.link(p, link)
    return d


jobs = [(k, 0) for k in args.folds]
if args.canonical_fold in args.folds:
    jobs += [(args.canonical_fold, s) for s in args.extra_seeds]

for test, seed in jobs:
    name = f"test{test}_seed{seed}"
    run_dir = runs_root / name
    val = (test - 1) % N_FOLDS
    cfg = dict(base, seed=seed, data_dir=str(fold_data_dir(test)), run_dir=str(run_dir))
    cfg["source"] = dict(base["source"], folds={"train": [k for k in range(N_FOLDS) if k not in (test, val)],
                                               "val": [val], "test": [test]})
    cfg_path = write_cfg(run_dir / "config.yaml", cfg)
    print(f"\n===== {name} =====", flush=True)

    if not (run_dir / "index_thresholds.json").exists():
        run("scripts/fit_index.py", cfg_path)
    for v in VARIANTS:
        if not (run_dir / v / "train_log.json").exists():
            run("scripts/train.py", cfg_path, v)
    if not (run_dir / "eval" / "overall.csv").exists():
        run("scripts/evaluate.py", cfg_path)
    if test == args.canonical_fold and seed == 0 and not args.no_explain:
        if not (run_dir / "explain" / "summary.json").exists():
            run("scripts/explain.py", cfg_path)

run("scripts/aggregate_cv.py", runs_root)
