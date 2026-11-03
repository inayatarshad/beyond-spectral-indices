"""Tune the global index threshold on the train split: python scripts/fit_index.py configs/burn.yaml"""

import argparse
import json

from bsi.config import load_config
from bsi.data.chips import chip_paths, load_chip
from bsi.indices import compute_index, tune_threshold

parser = argparse.ArgumentParser()
parser.add_argument("config")
cfg = load_config(parser.parse_args().config)

values, masks = [], []
for p in chip_paths(cfg["data_dir"], "train"):
    chip = load_chip(p)
    values.append(compute_index(chip["image"], cfg["task"], cfg["index"]["name"], cfg["use_pre"]))
    masks.append(chip["mask"])

result = {
    "index": cfg["index"]["name"],
    "literature": cfg["index"]["literature_threshold"],
    "tuned": tune_threshold(values, masks, seed=cfg["seed"]),
}
(cfg["run_dir"] / "index_thresholds.json").write_text(json.dumps(result, indent=2))
print(result)
