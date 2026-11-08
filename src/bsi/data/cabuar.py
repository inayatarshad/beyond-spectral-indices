"""CaBuAr (California Burned Areas) -> chips.

The HDF5 layout is discovered rather than hard-coded: any group holding both a
"post_fire" and a "mask" dataset is one event, and its fold comes from a "fold"
attribute on that group or a parent. Run `prepare_data.py --inspect` first and
adjust here if the published layout differs.
"""

import re
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the BZip2 filter CaBuAr is compressed with
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from .chips import IGNORE, save_chip

# Patch comments the CaBuAr authors discard in their own experiments (Rege Cambrin et al., 2023).
DISCARDED_COMMENTS = {4, 8, 12}


def download(hf_repo: str, hf_file: str | None, local_file: str | None = None) -> Path:
    if local_file and Path(local_file).exists():
        return Path(local_file)
    if hf_file is None:
        files = [f for f in HfApi().list_repo_files(hf_repo, repo_type="dataset") if f.endswith(".hdf5")]
        if not files:
            raise FileNotFoundError(f"no .hdf5 file in {hf_repo}")
        preferred = [f for f in files if "512" in f]
        hf_file = (preferred or files)[0]
        print(f"using {hf_file} (available: {files})")
    return Path(hf_hub_download(hf_repo, hf_file, repo_type="dataset"))


def inspect(path: Path, limit: int = 40) -> None:
    shown = 0

    def visit(name, obj):
        nonlocal shown
        if shown < limit:
            desc = f"dataset {obj.shape} {obj.dtype}" if isinstance(obj, h5py.Dataset) else "group"
            print(f"{name}: {desc} attrs={dict(obj.attrs)}")
            shown += 1

    with h5py.File(path, "r") as f:
        print(f"root attrs={dict(f.attrs)}")
        f.visititems(visit)


def _to_chw(a: np.ndarray) -> np.ndarray:
    """Images are stored HWC in some releases and CHW in others."""
    if a.ndim == 3 and a.shape[-1] <= 13 and a.shape[0] > 13:
        a = np.moveaxis(a, -1, 0)
    return a


def _fold(obj) -> int | None:
    """Fold id from a "fold" attribute or a "fold<N>" group name on the path."""
    while obj is not None:
        if "fold" in obj.attrs:
            return int(obj.attrs["fold"])
        m = re.fullmatch(r"fold_?(\d+)", obj.name.rsplit("/", 1)[-1])
        if m:
            return int(m.group(1))
        obj = obj.parent if obj.name != "/" else None
    return None


def _events(f: h5py.File):
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "post_fire" in obj and "mask" in obj:
            found.append(obj)

    f.visititems(visit)
    return found


def prepare(cfg: dict) -> None:
    src = cfg["source"]
    path = download(src["hf_repo"], src["hf_file"], src.get("local_file"))
    fold_to_split = {fold: split for split, folds in src["folds"].items() for fold in folds}
    size = src["chip_size"]
    counts = {"train": 0, "val": 0, "test": 0, "skipped_no_pre": 0, "skipped_fold": 0, "skipped_comment": 0}

    with h5py.File(path, "r") as f:
        for ev in _events(f):
            fold = _fold(ev)
            split = fold_to_split.get(fold)
            if split is None:
                counts["skipped_fold"] += 1
                continue
            if DISCARDED_COMMENTS & set(np.atleast_1d(ev.attrs.get("comments", [])).tolist()):
                counts["skipped_comment"] += 1
                continue
            post = _to_chw(ev["post_fire"][()]).astype(np.float32)
            if cfg["use_pre"]:
                if "pre_fire" not in ev:
                    counts["skipped_no_pre"] += 1
                    continue
                pre = _to_chw(ev["pre_fire"][()]).astype(np.float32)
                image = np.concatenate([post, pre], 0)
            else:
                image = post
            if image.max() > 2.0:  # stored as L2A digital numbers
                image = image / 10000.0
            mask = np.squeeze(ev["mask"][()]).astype(np.uint8)
            mask = np.where(mask > 0, 1, 0).astype(np.uint8)
            # all-zero pixels are outside the swath
            mask[np.all(image == 0, axis=0)] = IGNORE

            uid = ev.name.strip("/").replace("/", "_")
            h, w = mask.shape
            for r in range(0, h - size + 1, size):
                for c in range(0, w - size + 1, size):
                    m = mask[r : r + size, c : c + size]
                    if (m == IGNORE).mean() > 0.5:
                        continue
                    save_chip(cfg["data_dir"], split, f"{uid}_{r}_{c}", image[:, r : r + size, c : c + size], m)
                    counts[split] = counts.get(split, 0) + 1  # split names are free-form (e.g. fold0..fold4)
    print(counts)
