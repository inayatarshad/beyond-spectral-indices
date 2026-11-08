import os
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a task config; BSI_ROOT, if set, relocates data_dir and run_dir (e.g. to a larger disk)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    root = Path(os.environ.get("BSI_ROOT", "."))
    cfg["data_dir"] = root / cfg["data_dir"]
    cfg["run_dir"] = root / cfg["run_dir"]
    if cfg.get("source", {}).get("local_file"):
        cfg["source"]["local_file"] = str(root / cfg["source"]["local_file"])
    cfg["run_dir"].mkdir(parents=True, exist_ok=True)
    return cfg
