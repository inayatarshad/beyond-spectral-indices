"""Download a source dataset and cut it into chips: python scripts/prepare_data.py configs/burn.yaml"""

import argparse

from bsi.config import load_config
from bsi.data.chips import compute_band_stats

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--inspect", action="store_true", help="print the raw file layout and stop (CaBuAr only)")
args = parser.parse_args()
cfg = load_config(args.config)

if cfg["source"]["name"] == "cabuar":
    from bsi.data import cabuar

    if args.inspect:
        cabuar.inspect(cabuar.download(cfg["source"]["hf_repo"], cfg["source"]["hf_file"]))
        raise SystemExit
    cabuar.prepare(cfg)
else:
    from bsi.data import planetary

    planetary.prepare(cfg)

stats = compute_band_stats(cfg["data_dir"])
print("band means:", stats["mean"].round(4))
