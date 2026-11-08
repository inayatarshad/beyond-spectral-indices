"""Turn eval/explain outputs into the paper's tables, figures and result macros.

python scripts/make_paper_assets.py configs/burn.yaml configs/forest.yaml

Every number in paper/tables and every panel in paper/figures comes from this
script, so the manuscript never contains a hand-typed result.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from bsi.bands import S2_BANDS
from bsi.config import load_config
from bsi.data.chips import IGNORE, chip_paths, compute_band_stats
from bsi.predict import Predictor

PAPER = Path("paper")
TABLES, FIGS = PAPER / "tables", PAPER / "figures"
COL, DCOL = 3.5, 7.16  # IEEE single / double column width, inches
METHODS = ["index_literature", "index_otsu", "index_tuned", "lgbm_index", "lgbm_all", "unet_index", "unet_all"]
LABEL = {"index_literature": "Index, literature $\\tau$", "index_otsu": "Index, Otsu $\\tau$",
         "index_tuned": "Index, tuned $\\tau$", "lgbm_index": "GBDT, index bands", "lgbm_all": "GBDT, all bands",
         "unet_index": "U-Net, index bands", "unet_all": "U-Net, all bands"}
TEX_LABEL = {k: v.replace("$\\tau$", "$\\tau$") for k, v in LABEL.items()}
# Okabe-Ito colour-blind-safe palette
GREY, BLUE, ORANGE, GREEN, VERM, SKY, PURPLE = "#8c8c8c", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"
METHOD_COLOR = {"index_literature": "#c7c7c7", "index_otsu": "#a6a6a6", "index_tuned": GREY,
                "lgbm_index": SKY, "lgbm_all": BLUE, "unet_index": ORANGE, "unet_all": VERM}
BEST = "unet_index"  # converged reference model for the stratified and qualitative analysis
TASK_NAME = {"burn": "Burn scars (CaBuAr)", "forest": "Tree cover (S2 + WorldCover)"}
INDEX_NAME = {"burn": "dNBR", "forest": "NDVI"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def overall(cfg):
    return pd.read_csv(cfg["run_dir"] / "eval" / "overall.csv").set_index("method")


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------- tables

def main_table(cfgs):
    frames = {c["task"]: overall(c) for c in cfgs}
    best = {t: df["iou"].max() for t, df in frames.items()}
    head = " & ".join(rf"\multicolumn{{3}}{{c}}{{{TASK_NAME[t].split(' (')[0]}}}" for t in frames)
    sub = " & ".join(["IoU [95\\% CI] & F1 & Prec./Rec."] * len(frames))
    lines = [r"\begin{tabular}{@{}l" + "ccc" * len(frames) + "@{}}", r"\toprule", f"Method & {head} \\\\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(frames))),
             f" & {sub} \\\\", r"\midrule"]
    for m in METHODS:
        cells = []
        for t, df in frames.items():
            if m not in df.index:
                cells += ["--"] * 3
                continue
            r = df.loc[m]
            iou = f"{r.iou:.3f}"
            if np.isclose(r.iou, best[t]):
                iou = rf"\textbf{{{iou}}}"
            cells += [rf"{iou} {{\scriptsize[{r.iou_lo:.3f}, {r.iou_hi:.3f}]}}", f"{r.f1:.3f}",
                      f"{r.precision:.2f}/{r.recall:.2f}"]
        lines.append(f"{TEX_LABEL[m]} & " + " & ".join(cells) + r" \\")
        if m == "index_tuned":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "main_results.tex").write_text("\n".join(lines))


def decomposition(cfg):
    """IoU gains along both paths of the 2x2 design."""
    df = overall(cfg)["iou"]
    g = lambda a, b: df.get(b, np.nan) - df.get(a, np.nan)
    return {
        "rule": g("index_tuned", "lgbm_index"),
        "spectrum_pixel": g("lgbm_index", "lgbm_all"),
        "space_after_spectrum": g("lgbm_all", "unet_all"),
        "space_pixel_bands": g("lgbm_index", "unet_index"),
        "spectrum_after_space": g("unet_index", "unet_all"),
        "total": g("index_tuned", "unet_all"),
        "threshold": g("index_literature", "index_tuned"),
    }


def decomposition_table(cfgs):
    rows = [("Threshold calibration", "literature $\\to$ tuned $\\tau$", "threshold"),
            ("Learned rule", "index $\\to$ GBDT, index bands", "rule"),
            ("Spectrum (no context)", "GBDT index $\\to$ GBDT all", "spectrum_pixel"),
            ("Space (after spectrum)", "GBDT all $\\to$ U-Net all", "space_after_spectrum"),
            ("Space (index bands)", "GBDT index $\\to$ U-Net index", "space_pixel_bands"),
            ("Spectrum (with context)", "U-Net index $\\to$ U-Net all", "spectrum_after_space"),
            ("Total learned gain", "tuned index $\\to$ U-Net all", "total")]
    dec = {c["task"]: decomposition(c) for c in cfgs}
    lines = [r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             "Effect & Step & " + " & ".join(rf"$\Delta$IoU {t}" for t in dec) + r" \\", r"\midrule"]
    for name, step, key in rows:
        if key == "total":
            lines.append(r"\midrule")
        lines.append(f"{name} & {step} & " + " & ".join(f"{d[key]:+.3f}" for d in dec.values()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "decomposition.tex").write_text("\n".join(lines).replace("rrr", "rr" if len(dec) == 1 else "rr"))
    return dec


def gain_table(cfg, family, order=None, title=None):
    g = pd.read_csv(cfg["run_dir"] / "eval" / "gain.csv")
    s = pd.read_csv(cfg["run_dir"] / "eval" / "strata.csv")
    g = g[(g.family == family) & (g.method == BEST)].set_index("stratum")
    s = s[s.family == family]
    if g.empty:
        return
    idx = [o for o in (order or g.index) if o in g.index]
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Stratum & Pixels & Err.\ index & Err.\ U-Net & Gain share & Lift \\",
             r" & (\%) & (\%) & (\%) & (\%) & \\", r"\midrule"]
    for st in idx:
        r = g.loc[st]
        ei = s[(s.method == "index_tuned") & (s.stratum == st)].error_rate.iloc[0]
        eu = s[(s.method == BEST) & (s.stratum == st)].error_rate.iloc[0]
        lift = r.reduction_share / r.pixel_share if r.pixel_share > 0 else np.nan
        lines.append(f"{str(st).replace('_', ' ')} & {100 * r.pixel_share:.1f} & {100 * ei:.1f} & {100 * eu:.1f} & "
                     f"{100 * r.reduction_share:.1f} & {lift:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"gain_{cfg['task']}_{family}.tex").write_text("\n".join(lines))


def faithfulness_table(cfgs):
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Task / model & IG vs.\ occl. & IG vs.\ GradSHAP & IG vs.\ random & $r_{50}$ / $r_{90}$ (m) \\",
             r"\midrule"]
    for cfg in cfgs:
        sj = cfg["run_dir"] / "explain" / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        for v in ("unet_index", "unet_all"):
            if f"{v}_ig_vs_occlusion_spearman" not in s:
                continue
            lines.append(
                f"{cfg['task'].capitalize()} / {LABEL[v]} & {s[f'{v}_ig_vs_occlusion_spearman']:.2f} & "
                f"{s.get(f'{v}_ig_vs_gradshap_spearman', np.nan):.2f} & {s[f'{v}_randomisation_spearman']:.2f} & "
                f"{cfg['pixel_size_m'] * s[f'{v}_context_r50_px_median']:.0f} / "
                f"{cfg['pixel_size_m'] * s[f'{v}_context_r90_px_median']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "faithfulness.tex").write_text("\n".join(lines))


def data_table(cfgs):
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Task & GSD (m) & Train & Val. & Test & Target px (\%) \\", r"\midrule"]
    for cfg in cfgs:
        n = {sp: len(chip_paths(cfg["data_dir"], sp)) for sp in ("train", "val", "test")}
        df = overall(cfg)
        pos = df.loc["index_tuned", "positives"] / df.loc["index_tuned", "pixels"]
        lines.append(f"{TASK_NAME[cfg['task']].split(' (')[0]} & {cfg['pixel_size_m']} & {n['train']} & {n['val']} & "
                     f"{n['test']} & {100 * pos:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "data.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------- figures

def ladder_figure(cfgs):
    fig, axes = plt.subplots(1, len(cfgs), figsize=(DCOL, 1.9), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        df = overall(cfg)
        ms = [m for m in METHODS if m in df.index]
        y = np.arange(len(ms))[::-1]
        err = np.array([df.loc[ms, "iou"] - df.loc[ms, "iou_lo"], df.loc[ms, "iou_hi"] - df.loc[ms, "iou"]])
        ax.barh(y, df.loc[ms, "iou"], xerr=err, color=[METHOD_COLOR[m] for m in ms], height=0.7,
                error_kw={"elinewidth": 0.6, "capsize": 1.5})
        for yi, m in zip(y, ms):
            ax.text(df.loc[m, "iou_hi"] + 0.01, yi, f"{df.loc[m, 'iou']:.3f}", va="center", fontsize=6.5)
        ax.axvline(df.loc["index_tuned", "iou"], color=GREY, lw=0.6, ls=":")
        ax.set_yticks(y, [LABEL[m].replace("Index", INDEX_NAME[cfg["task"]]) for m in ms])
        ax.set_xlim(0, min(1.0, df["iou_hi"].max() + 0.12))
        ax.set_xlabel("Test IoU")
        ax.set_title(TASK_NAME[cfg["task"]])
    fig.tight_layout(w_pad=2.5)
    save(fig, "ladder")


def square_figure(cfgs, decs):
    """The 2x2 design as annotated IoU matrices."""
    fig, axes = plt.subplots(1, len(cfgs), figsize=(COL, 1.55), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        df = overall(cfg)["iou"]
        mat = np.array([[df.get("lgbm_index", np.nan), df.get("unet_index", np.nan)],
                        [df.get("lgbm_all", np.nan), df.get("unet_all", np.nan)]])
        ax.imshow(mat, cmap="Blues", vmin=np.nanmin(mat) - 0.15, vmax=np.nanmax(mat) + 0.02)
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > np.nanmean(mat) else "black")
        ax.set_xticks([0, 1], ["per pixel", "spatial"])
        ax.set_yticks([0, 1], ["index\nbands", "all\nbands"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{cfg['task'].capitalize()} (tuned {INDEX_NAME[cfg['task']]}: {df['index_tuned']:.3f})", fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "square")


def gain_figure(cfgs):
    """Pixel share vs. share of the U-Net's error reduction, per stratum."""
    fams = {"burn": ("severity", ["regrowth", "unburned", "low", "mod_low", "mod_high", "high"]),
            "forest": ("landcover", None)}
    fig, axes = plt.subplots(1, len(cfgs), figsize=(DCOL, 1.75), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        fam, order = fams[cfg["task"]]
        g = pd.read_csv(cfg["run_dir"] / "eval" / "gain.csv")
        g = g[(g.family == fam) & (g.method == BEST)].set_index("stratum")
        g = g[g.pixel_share > 0.002]
        idx = [o for o in order if o in g.index] if order else g.sort_values("pixel_share", ascending=False).index
        x = np.arange(len(idx))
        ax.bar(x - 0.19, 100 * g.loc[idx, "pixel_share"], 0.38, color="#bdbdbd", label="share of pixels")
        ax.bar(x + 0.19, 100 * g.loc[idx, "reduction_share"], 0.38, color=VERM, label="share of error reduction")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x, [str(i).replace("_", "-") for i in idx], rotation=0)
        ax.set_ylabel("%")
        ax.set_title(f"{TASK_NAME[cfg['task']]}: by {'dNBR severity' if fam == 'severity' else 'WorldCover class'}")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "gain")


def band_figure(cfgs):
    fig, axes = plt.subplots(len(cfgs), 1, figsize=(COL, 1.45 * len(cfgs) + 0.3), squeeze=False, sharex=True)
    for ax, cfg in zip(axes[:, 0], cfgs):
        ex = cfg["run_dir"] / "explain"
        attr, shp, occ = (pd.read_csv(ex / f) for f in ("band_attribution.csv", "tree_shap.csv", "occlusion.csv"))
        a = attr[(attr.variant == "unet_all") & (attr.state == "trained") & (attr.attribution == "ig")]
        s = shp[(shp.variant == "lgbm_all") & (shp.pixels == "all")]
        o = occ[(occ.method == "unet_all") & occ.group.isin(S2_BANDS)].set_index("group")
        views = [("U-Net |IG|", BLUE, [a[a.channel.str.endswith(b)].share.sum() for b in S2_BANDS]),
                 ("GBDT |SHAP|", ORANGE, [s[s.channel.str.endswith(b)].mean_abs_shap.sum() for b in S2_BANDS]),
                 ("U-Net occlusion", GREEN, [max(o.loc[b, "iou_drop"], 0) if b in o.index else 0 for b in S2_BANDS])]
        x = np.arange(len(S2_BANDS))
        for k, (name, color, v) in enumerate(views):
            v = np.asarray(v, float)
            ax.bar(x + (k - 1) * 0.27, v / (v.max() + 1e-12), 0.27, color=color, label=name)
        idx_bands = {"burn": ["B08", "B12"], "forest": ["B04", "B08"]}[cfg["task"]]
        for b in idx_bands:
            ax.axvspan(S2_BANDS.index(b) - 0.45, S2_BANDS.index(b) + 0.45, color="#000000", alpha=0.06, lw=0)
        ax.set_ylabel("rel. importance")
        ax.set_title(f"{TASK_NAME[cfg['task']]}  (shaded: bands used by {INDEX_NAME[cfg['task']]})", fontsize=7)
        ax.set_ylim(0, 1.08)
    axes[-1, 0].set_xticks(x, S2_BANDS, rotation=0, fontsize=6)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    fig.tight_layout(h_pad=0.6)
    save(fig, "bands")


def context_figure(cfgs):
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    styles = {"unet_all": "-", "unet_index": "--"}
    colors = {"burn": VERM, "forest": GREEN}
    for cfg in cfgs:
        ctx = pd.read_csv(cfg["run_dir"] / "explain" / "context.csv")
        for v, g in ctx.groupby("variant"):
            ax.plot(cfg["pixel_size_m"] * g.radius_px, g.mass_share, styles[v], color=colors[cfg["task"]], lw=1.0,
                    label=f"{cfg['task']}, {LABEL[v].split(', ')[1]}")
    for q in (0.5, 0.9):
        ax.axhline(q, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("Distance from target pixel (m)")
    ax.set_ylabel("Cumulative |IG| mass")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 640)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "context")


def _overlay(pred, mask):
    """0 TN, 1 TP, 2 FP, 3 FN, 4 no label."""
    o = np.zeros(mask.shape, np.uint8)
    t = mask == 1
    o[pred & t], o[pred & ~t], o[~pred & t] = 1, 2, 3
    o[mask == IGNORE] = 4
    return o


ERR_CMAP = ListedColormap(["#f2f2f2", "#1b9e77", "#d95f02", "#7570b3", "#ffffff"])


def _rgb(img):
    rgb = np.stack([img[3], img[2], img[1]], -1)  # B04, B03, B02 (post-fire for burn chips)
    lo, hi = np.percentile(rgb, [2, 98])
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) ** 0.8


def examples_figure(cfgs, n=3):
    rows = []
    for cfg in cfgs:
        pc = pd.read_csv(cfg["run_dir"] / "eval" / "per_chip.csv")
        pc["iou"] = pc.tp / (pc.tp + pc.fp + pc.fn).clip(lower=1)
        pc["pos"] = pc.tp + pc.fn
        w = pc.pivot(index="chip", columns="method", values="iou")
        pos = pc[pc.method == BEST].set_index("chip").pos
        w = w[pos.loc[w.index] > 2000]
        top = (w[BEST] - w.index_tuned).sort_values(ascending=False).index[:n]
        pred = Predictor(cfg, ["index_tuned", "lgbm_all", BEST], compute_band_stats(cfg["data_dir"]))
        paths = {p.stem: p for p in chip_paths(cfg["data_dir"], "test")}
        for cid in top:
            chip, preds = pred.chip(paths[cid])
            rows.append((cfg["task"], chip, preds, w.loc[cid]))
    fig, axes = plt.subplots(len(rows), 5, figsize=(DCOL, 1.42 * len(rows)), squeeze=False)
    heads = ["Sentinel-2 RGB (post-fire)", "Reference", None, "GBDT, all bands", "U-Net, index bands"]
    for r, (task, chip, preds, ious) in enumerate(rows):
        panels = [_rgb(chip["image"]), _overlay(chip["mask"] == 1, chip["mask"]),
                  _overlay(preds["index_tuned"], chip["mask"]), _overlay(preds["lgbm_all"], chip["mask"]),
                  _overlay(preds[BEST], chip["mask"])]
        sub = ["", "", f"IoU {ious.index_tuned:.2f}", f"IoU {ious.lgbm_all:.2f}", f"IoU {ious[BEST]:.2f}"]
        for c, (ax, im) in enumerate(zip(axes[r], panels)):
            ax.imshow(im, cmap=None if im.ndim == 3 else ERR_CMAP, vmin=0, vmax=4, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(0.4)
            if r == 0 or rows[r - 1][0] != task:
                ax.set_title(heads[c] or f"Tuned {INDEX_NAME[task]}", fontsize=7)
            if sub[c]:
                ax.text(0.03, 0.04, sub[c], transform=ax.transAxes, fontsize=6.5,
                        bbox={"facecolor": "white", "alpha": 0.8, "pad": 1, "lw": 0})
        axes[r][0].set_ylabel(task, fontsize=7)
    fig.legend(handles=[Patch(color="#1b9e77", label="true positive"), Patch(color="#d95f02", label="false positive"),
                        Patch(color="#7570b3", label="false negative")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(h_pad=0.3, w_pad=0.3, rect=(0, 0.03, 1, 1))
    save(fig, "examples")


def study_figure(cfg_forest):
    """One RGB thumbnail per forest region with its tree-cover fraction."""
    paths = chip_paths(cfg_forest["data_dir"], "test")
    by_aoi = {}
    for p in paths:
        aoi = p.stem.rsplit("_", 2)[0]
        by_aoi.setdefault(aoi, []).append(p)
    aois = list(cfg_forest["source"]["aois"])
    fig, axes = plt.subplots(2, len(aois), figsize=(DCOL, 2.25), squeeze=False)
    from bsi.data.chips import load_chip

    for j, aoi in enumerate(aois):
        cands = by_aoi.get(aoi, [])
        if not cands:
            for ax in axes[:, j]:
                ax.axis("off")
            continue
        chips = [load_chip(p) for p in cands[:12]]
        frac = [((c["mask"] == 1).mean()) for c in chips]
        chip = chips[int(np.argsort(frac)[len(frac) // 2])]
        axes[0, j].imshow(_rgb(chip["image"]))
        axes[1, j].imshow(np.where(chip["mask"] == IGNORE, 0.5, chip["mask"]), cmap="Greens", vmin=0, vmax=1.3)
        name = aoi.replace("_", " ").title()
        axes[0, j].set_title(name, fontsize=6.5)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("RGB", fontsize=7)
    axes[1, 0].set_ylabel("tree cover", fontsize=7)
    fig.tight_layout(w_pad=0.2, h_pad=0.2)
    save(fig, "regions")


def macros(cfgs, decs):
    lines = []

    def emit(name, value, fmt):
        name = "".join(ch for ch in name if ch.isalpha())
        lines.append(rf"\newcommand{{\{name}}}{{{format(value, fmt)}}}")

    def camel(s):
        s = s.replace("50", "fifty").replace("90", "ninety")
        return "".join(w.capitalize() for w in s.split("_"))

    for cfg in cfgs:
        t = cfg["task"].capitalize()
        df = overall(cfg)
        for m in df.index:
            emit(f"iou{t}{camel(m)}", df.loc[m, "iou"], ".3f")
            emit(f"fone{t}{camel(m)}", df.loc[m, "f1"], ".3f")
            emit(f"lo{t}{camel(m)}", df.loc[m, "iou_lo"], ".3f")
            emit(f"hi{t}{camel(m)}", df.loc[m, "iou_hi"], ".3f")
        for k, v in decs[cfg["task"]].items():
            emit(f"gain{t}{camel(k)}", v, ".3f")
        sj = cfg["run_dir"] / "explain" / "summary.json"
        if sj.exists():
            for k, v in json.loads(sj.read_text()).items():
                emit(f"res{t}{camel(k)}", v, ".2f")
                if "context_r" in k:
                    emit(f"res{t}{camel(k)}M", cfg["pixel_size_m"] * v, ".0f")
    (TABLES / "results_macros.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    cfgs = [load_config(c) for c in parser.parse_args().configs]
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    main_table(cfgs)
    decs = decomposition_table(cfgs)
    data_table(cfgs)
    faithfulness_table(cfgs)
    for cfg in cfgs:
        for fam in ("severity", "landcover", "index_margin", "boundary", "ndvi", "ndvi_pre"):
            order = {"severity": ["regrowth", "unburned", "low", "mod_low", "mod_high", "high"],
                     "index_margin": ["ambiguous", "near", "clear"], "boundary": ["edge", "interior"]}.get(fam)
            gain_table(cfg, fam, order)
    ladder_figure(cfgs)
    gain_figure(cfgs)
    context_figure(cfgs)
    band_figure(cfgs)
    examples_figure(cfgs)
    forest = [c for c in cfgs if c["task"] == "forest"]
    if forest:
        study_figure(forest[0])
    macros(cfgs, decs)
    print(f"wrote {TABLES} and {FIGS}")
