"""Per-pixel strata for the "when" analysis.

Each family maps a chip to a string label per pixel; errors are then pooled
per (family, label, method).
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from .bands import channel_index, channel_names
from .indices import compute_index

WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
              70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
# USGS dNBR severity classes (Key & Benson 2006)
DNBR_BINS = [-np.inf, -0.10, 0.10, 0.27, 0.44, 0.66, np.inf]
DNBR_LABELS = ["regrowth", "unburned", "low", "mod_low", "mod_high", "high"]
NDVI_BINS = [-np.inf, 0.2, 0.4, 0.6, np.inf]
NDVI_LABELS = ["ndvi<0.2", "0.2-0.4", "0.4-0.6", "ndvi>0.6"]
EDGE_PX = 2


def _bin(values, bins, labels):
    return np.array(labels, dtype=object)[np.clip(np.digitize(values, bins) - 1, 0, len(labels) - 1)]


def boundary(mask: np.ndarray) -> np.ndarray:
    target = mask == 1
    dist = np.where(target, distance_transform_edt(target), distance_transform_edt(~target))
    return np.where(dist <= EDGE_PX, "edge", "interior").astype(object)


def index_margin(image, cfg, tuned_threshold) -> np.ndarray:
    idx = compute_index(image, cfg["task"], cfg["index"]["name"], cfg["use_pre"])
    m = np.abs(idx - tuned_threshold)
    return _bin(m, [-np.inf, 0.05, 0.15, np.inf], ["ambiguous", "near", "clear"])


def chip_strata(chip: dict, cfg: dict, tuned_threshold: float) -> dict[str, np.ndarray]:
    image, task = chip["image"], cfg["task"]
    names = channel_names(task, cfg["use_pre"])
    s = {"boundary": boundary(chip["mask"]), "index_margin": index_margin(image, cfg, tuned_threshold)}

    pre = "pre_" if task == "burn" and cfg["use_pre"] else ("post_" if task == "burn" else "")
    red, nir = image[channel_index(names, "B04", pre)], image[channel_index(names, "B08", pre)]
    ndvi = (nir - red) / (nir + red + 1e-6)
    s["ndvi_pre" if task == "burn" else "ndvi"] = _bin(ndvi, NDVI_BINS, NDVI_LABELS)

    if task == "burn" and cfg["use_pre"]:
        s["severity"] = _bin(compute_index(image, "burn", "dnbr", True), DNBR_BINS, DNBR_LABELS)
    if "worldcover" in chip:
        lut = np.array([WORLDCOVER.get(i, "nodata") for i in range(256)], dtype=object)
        s["landcover"] = lut[chip["worldcover"]]
    return s
