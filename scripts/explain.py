"""The "why" analysis: python scripts/explain.py configs/burn.yaml

Writes to runs/<task>/explain/:
  band_attribution.csv   per-channel share of |IG| and |GradientSHAP| for each U-Net (trained and randomised)
  context.csv            cumulative attribution mass vs distance from the target pixel
  occlusion.csv          IoU drop when a band group is replaced by its train mean, every method
  tree_shap.csv          mean |SHAP| per feature for each LightGBM rung, all pixels and index-ambiguous pixels
  summary.json           scalar results quoted in the paper
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bsi.bands import S2_BANDS, channel_names
from bsi.config import load_config
from bsi.data.chips import IGNORE, chip_paths, compute_band_stats, load_chip
from bsi.explain import band_attribution, context_profile, occlusion, radius_at, rank_agreement, tree_shap
from bsi.models import build_unet, channels_for, load_lgbm, load_unet
from bsi.predict import ALL_METHODS, Predictor, load_thresholds
from bsi.strata import index_margin

parser = argparse.ArgumentParser()
parser.add_argument("config")
cfg = load_config(parser.parse_args().config)
out = cfg["run_dir"] / "explain"
out.mkdir(exist_ok=True)
ex = cfg["explain"]
rng = np.random.default_rng(cfg["seed"])
torch.manual_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
stats = compute_band_stats(cfg["data_dir"])
mean, std = stats["mean"], stats["std"]
names = channel_names(cfg["task"], cfg["use_pre"])
tuned = load_thresholds(cfg)["tuned"]
summary = {}

test_paths = chip_paths(cfg["data_dir"], "test")
chips = [load_chip(p) for p in rng.choice(test_paths, min(ex["n_chips"], len(test_paths)), replace=False)]
chips = [c for c in chips if (c["mask"] == 1).any()]
train_paths = chip_paths(cfg["data_dir"], "train")
shap_bg = torch.stack([torch.from_numpy((load_chip(p)["image"] - mean[:, None, None]) / std[:, None, None])
                       for p in rng.choice(train_paths, 8, replace=False)]).float().to(device)


def norm(img):
    return torch.from_numpy((img - mean[:, None, None]) / std[:, None, None]).float()[None].to(device)


def exists(m):
    return (cfg["run_dir"] / m / ("model.pt" if m.startswith("unet") else "model.txt")).exists()


# band groups: one per Sentinel-2 band (post and pre together), plus the whole pre/post stacks
groups = {b: [i for i, n in enumerate(names) if n.endswith(b)] for b in S2_BANDS}
if cfg["task"] == "burn" and cfg["use_pre"]:
    groups["all_post"] = [i for i, n in enumerate(names) if n.startswith("post_")]
    groups["all_pre"] = [i for i, n in enumerate(names) if n.startswith("pre_")]

# 1) gradient attributions and the randomisation sanity check (U-Nets)
attr_rows, ctx_rows = [], []
for variant in [m for m in ("unet_index", "unet_all") if exists(m)]:
    trained = load_unet(cfg, variant, device)
    random_net = build_unet(cfg, variant, weights=False).to(device).eval()
    shares = {"trained": [], "random": []}
    curves = []
    for k, chip in enumerate(tqdm(chips, desc=f"attribution {variant}")):
        x = norm(chip["image"])
        with torch.no_grad():
            region = trained(x)[0] > 0
        if region.sum() == 0:
            region = torch.from_numpy(chip["mask"] == 1).to(device)
        shares["trained"].append(band_attribution(trained, x, region, ex["ig_steps"], shap_bg))
        if k < 10:
            shares["random"].append(band_attribution(random_net, x, region, ex["ig_steps"]))
        valid = np.argwhere(chip["mask"] != IGNORE)
        pick = valid[rng.choice(len(valid), min(ex["context_pixels_per_chip"], len(valid)), replace=False)]
        curves.append(context_profile(trained, x, [tuple(p) for p in pick], ex["ig_steps"]))
    for state, lst in shares.items():
        for method in lst[0]:
            v = np.mean([s[method] for s in lst], 0)
            for i, n in enumerate(names):
                attr_rows.append({"variant": variant, "state": state, "attribution": method, "channel": n, "share": v[i]})
    ig_tr = np.mean([s["ig"] for s in shares["trained"]], 0)
    ig_rd = np.mean([s["ig"] for s in shares["random"]], 0)
    summary[f"{variant}_randomisation_spearman"] = rank_agreement(ig_tr, ig_rd)
    if "gradshap" in shares["trained"][0]:
        gs_tr = np.mean([s["gradshap"] for s in shares["trained"]], 0)
        summary[f"{variant}_ig_vs_gradshap_spearman"] = rank_agreement(ig_tr, gs_tr)
    curves = np.concatenate(curves)
    for d, v in enumerate(curves.mean(0)):
        ctx_rows.append({"variant": variant, "radius_px": d, "mass_share": v})
    summary[f"{variant}_context_r50_px_median"] = float(np.median(radius_at(curves, 0.5)))
    summary[f"{variant}_context_r90_px_median"] = float(np.median(radius_at(curves, 0.9)))
    summary[f"{variant}_context_self_share_mean"] = float(curves[:, 0].mean())
pd.DataFrame(attr_rows).to_csv(out / "band_attribution.csv", index=False)
pd.DataFrame(ctx_rows).to_csv(out / "context.csv", index=False)

# 2) occlusion for every method
occ_rows = []
for m in [m for m in ALL_METHODS if m.startswith("index") or exists(m)]:
    pred = Predictor(cfg, [m], stats, device)
    drops = occlusion(lambda im: pred(im)[m], chips, groups, mean)
    for g, d in drops.items():
        occ_rows.append({"method": m, "group": g, "iou_drop": d})
occ = pd.DataFrame(occ_rows)
occ.to_csv(out / "occlusion.csv", index=False)

# faithfulness: do gradient attributions rank bands the way occlusion does?
attr = pd.DataFrame(attr_rows)
for variant in attr["variant"].unique() if len(attr) else []:
    a = attr[(attr.variant == variant) & (attr.state == "trained") & (attr.attribution == "ig")]
    band_share = [a[a.channel.str.endswith(b)].share.sum() for b in S2_BANDS]
    o = occ[(occ.method == variant) & occ.group.isin(S2_BANDS)].set_index("group").loc[S2_BANDS, "iou_drop"]
    summary[f"{variant}_ig_vs_occlusion_spearman"] = rank_agreement(np.array(band_share), o.to_numpy())

# 3) TreeSHAP for the LightGBM rungs, on all pixels and on index-ambiguous pixels
shap_rows = []
for variant in [m for m in ("lgbm_index", "lgbm_all") if exists(m)]:
    booster = load_lgbm(cfg, variant)
    ch = channels_for(variant, cfg["task"], cfg["use_pre"])
    feats, amb = [], []
    for chip in chips:
        valid = np.flatnonzero((chip["mask"] != IGNORE).ravel())
        pick = rng.choice(valid, min(len(valid), ex["shap_pixels"] // len(chips) + 1), replace=False)
        feats.append(chip["image"][ch].reshape(len(ch), -1)[:, pick].T)
        amb.append(index_margin(chip["image"], cfg, tuned).ravel()[pick] == "ambiguous")
    feats, amb = np.concatenate(feats), np.concatenate(amb)
    for subset, sel in (("all", np.ones(len(feats), bool)), ("ambiguous", amb)):
        if sel.sum() < 10:
            continue
        for i, v in enumerate(tree_shap(booster, feats[sel])):
            shap_rows.append({"variant": variant, "pixels": subset, "channel": names[ch[i]], "mean_abs_shap": v})
pd.DataFrame(shap_rows).to_csv(out / "tree_shap.csv", index=False)

(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
