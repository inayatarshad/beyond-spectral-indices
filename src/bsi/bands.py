"""Sentinel-2 L2A band bookkeeping.

All chips store bands in S2_BANDS order. Burn chips stack the post-fire image
first and the pre-fire image second, so channel names get a "post_"/"pre_" prefix.
"""

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# Bands each index uses; the "index-bands only" rung of the ladder restricts
# learned models to exactly these.
INDEX_BANDS = {
    "ndvi": ["B04", "B08"],  # (NIR - Red) / (NIR + Red)
    "nbr": ["B08", "B12"],   # (NIR - SWIR2) / (NIR + SWIR2)
}


def channel_names(task: str, use_pre: bool = True) -> list[str]:
    if task == "burn":
        names = [f"post_{b}" for b in S2_BANDS]
        if use_pre:
            names += [f"pre_{b}" for b in S2_BANDS]
        return names
    if task == "forest":
        return list(S2_BANDS)
    raise ValueError(f"unknown task {task!r}")


def channel_index(names: list[str], band: str, prefix: str = "") -> int:
    return names.index(f"{prefix}{band}")


def index_channel_subset(task: str, use_pre: bool = True) -> list[int]:
    """Channel ids that the reference index of a task can see."""
    names = channel_names(task, use_pre)
    if task == "burn":
        prefixes = ["post_", "pre_"] if use_pre else ["post_"]
        return [names.index(p + b) for p in prefixes for b in INDEX_BANDS["nbr"]]
    return [names.index(b) for b in INDEX_BANDS["ndvi"]]
