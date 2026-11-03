#!/usr/bin/env bash
# Full pipeline for one or both tasks: bash scripts/run_all.sh [burn] [forest]
set -euo pipefail
tasks=("$@")
[ ${#tasks[@]} -eq 0 ] && tasks=(burn forest)

for task in "${tasks[@]}"; do
  cfg="configs/${task}.yaml"
  python scripts/prepare_data.py "$cfg"
  python scripts/fit_index.py "$cfg"
  for variant in lgbm_index lgbm_all unet_index unet_all; do
    python scripts/train.py "$cfg" "$variant"
  done
  python scripts/evaluate.py "$cfg"
  python scripts/explain.py "$cfg"
done

python scripts/make_paper_assets.py $(for t in "${tasks[@]}"; do echo "configs/${t}.yaml"; done)
