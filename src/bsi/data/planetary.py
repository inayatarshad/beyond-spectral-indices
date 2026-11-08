"""Sentinel-2 composites + ESA WorldCover labels from Microsoft Planetary Computer -> chips."""

import os
import time

# transient blob-storage read failures are common over long downloads
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "8")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

import numpy as np
import planetary_computer
import pystac_client
import xarray as xr
from odc.stac import load as stac_load

from ..bands import S2_BANDS
from .chips import IGNORE, save_chip

CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
# SCL classes treated as invalid: no data, saturated, cloud shadow, cloud (medium/high), cirrus
SCL_INVALID = [0, 1, 3, 8, 9, 10]
WORLDCOVER_TREE = 10
MAX_SCENES = 8


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _composite(catalog, bbox, year, months, max_cloud, epsg):
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    items = sorted((it for it in items if it.datetime.month in months), key=lambda it: it.properties["eo:cloud_cover"])
    items = items[:MAX_SCENES]
    if not items:
        raise RuntimeError(f"no Sentinel-2 scenes for {bbox} in {year} months {months}")

    ds = stac_load(items, bands=S2_BANDS + ["SCL"], bbox=bbox, crs=f"EPSG:{epsg}", resolution=10,
                   groupby="solar_day", chunks={"x": 1024, "y": 1024})
    # Processing baseline >= 04.00 adds a +1000 DN offset to L2A reflectance.
    offset_by_day = {}
    for it in items:
        day = np.datetime64(it.datetime.date())
        baseline = it.properties.get("s2:processing_baseline", "00.00")
        offset_by_day[day] = 1000.0 if float(baseline) >= 4.0 else 0.0
    offsets = np.array([offset_by_day.get(np.datetime64(str(t)[:10]), 0.0) for t in ds.time.values])

    valid = ~ds["SCL"].isin(SCL_INVALID)
    bands = ds[S2_BANDS].to_array("band")
    bands = bands.where(valid & (bands > 0))
    bands = (bands - xr.DataArray(offsets, dims="time")) / 10000.0
    return bands.chunk({"time": -1}).median("time").compute()  # dask median needs time in one chunk


def _worldcover(catalog, like, bbox, year):
    items = catalog.search(collections=["esa-worldcover"], bbox=bbox,
                           datetime=f"{year}-01-01/{year}-12-31").item_collection()
    if not items:
        raise RuntimeError("no WorldCover tiles found")
    wc = stac_load(items, bands=["map"], like=like, resampling="nearest")["map"].max("time")
    return wc.values.astype(np.uint8)


def prepare(cfg: dict) -> None:
    src = cfg["source"]
    catalog = pystac_client.Client.open(CATALOG, modifier=planetary_computer.sign_inplace)
    size = src["chip_size"]
    fr = src["split_fractions"]
    counts = {"train": 0, "val": 0, "test": 0}

    raw_dir = cfg["data_dir"] / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, bbox in src["aois"].items():
        cache = raw_dir / f"{name}.npz"
        if cache.exists():  # composites take minutes to build; never fetch a region twice
            with np.load(cache) as z:
                composite, wc = z["composite"], z["worldcover"]
        else:
            lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            for attempt in range(3):
                try:
                    comp = _composite(catalog, bbox, src["year"], src["months"], src["max_cloud"], utm_epsg(lon, lat))
                    wc = _worldcover(catalog, comp, bbox, src["year"])
                    break
                except Exception as e:  # noqa: BLE001  retry with freshly signed URLs
                    if attempt == 2:
                        raise
                    print(f"{name}: attempt {attempt + 1} failed ({type(e).__name__}), retrying")
                    time.sleep(30)
            composite = comp.values.astype(np.float32)
            np.savez_compressed(cache, composite=composite, worldcover=wc)
        image = np.nan_to_num(composite, nan=0.0)
        mask = (wc == WORLDCOVER_TREE).astype(np.uint8)
        mask[(wc == 0) | np.isnan(composite).any(0)] = IGNORE

        h, w = mask.shape
        n_cols = w // size
        for ci, c in enumerate(range(0, w - size + 1, size)):
            pos = (ci + 0.5) / n_cols
            split = "train" if pos < fr["train"] else "val" if pos < fr["train"] + fr["val"] else "test"
            for r in range(0, h - size + 1, size):
                m = mask[r : r + size, c : c + size]
                if (m == IGNORE).mean() > 0.2:
                    continue
                save_chip(cfg["data_dir"], split, f"{name}_{r}_{c}", image[:, r : r + size, c : c + size], m,
                          worldcover=wc[r : r + size, c : c + size])
                counts[split] += 1
        print(name, counts)
