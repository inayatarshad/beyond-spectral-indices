from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = Path(cfg["data_dir"])
    cfg["run_dir"] = Path(cfg["run_dir"])
    cfg["run_dir"].mkdir(parents=True, exist_ok=True)
    return cfg
