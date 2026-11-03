"""Turn eval/explain outputs into the paper's tables and figures.

python scripts/make_paper_assets.py configs/burn.yaml configs/forest.yaml

Every number in paper/tables and paper/figures comes from this script, so the
paper never contains a hand-typed result.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bsi.bands import S2_BANDS
from bsi.config import load_config
from bsi.data.chips import chip_paths, compute_band_stats
from bsi.predict import Predictor

PAPER = Path("paper")
TABLES, FIGS = PAPER / "tables", PAPER / "figures"
LABEL = {"index_literature": "Index, literature thr.", "index_otsu": "Index, Otsu thr.",
         "index_tuned": "Index, tuned thr.", "lgbm_index": "LightGBM, index bands",
         "lgbm_all": "LightGBM, all bands", "unet_index": "U-Net, index bands", "unet_all": "U-Net, all bands"}
plt.rcParams.update({"font.size": 8, "figure.dpi": 200, "savefig.bbox": "tight"})


def main_table(cfgs):
    """IoU [95% CI] and F1 per method, one column group per task."""
    frames = {c["task"]: pd.read_csv(c["run_dir"] / "eval" / "overall.csv").set_index("method") for c in cfgs}
    lines = [r"\begin{tabular}{l" + "cc" * len(frames) + "}", r"\toprule",
             "Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t.capitalize()}}}" for t in frames) + r" \\",
             " & " + " & ".join(["IoU [95\\% CI] & F1"] * len(frames)) + r" \\", r"\midrule"]
    for m, label in LABEL.items():
        cells = []
        for df in frames.values():
            if m in df.index:
                r = df.loc[m]
                cells += [f"{r.iou:.3f} [{r.iou_lo:.3f}, {r.iou_hi:.3f}]", f"{r.f1:.3f}"]
            else:
                cells += ["--", "--"]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        if m == "index_tuned":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "main_results.tex").write_text("\n".join(lines))


def gain_table(cfg, family):
    g = pd.read_csv(cfg["run_dir"] / "eval" / "gain.csv")
    g = g[(g.family == family) & (g.method == "unet_all")]
    if g.empty:
        return
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Stratum & Pixels (\%) & Errors removed & Share of gain (\%) \\", r"\midrule"]
    for _, r in g.sort_values("reduction_share", ascending=False).iterrows():
        lines.append(f"{r.stratum.replace('_', ' ')} & {100 * r.pixel_share:.1f} & {r.error_reduction:,} & {100 * r.reduction_share:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / f"gain_{cfg['task']}_{family}.tex").write_text("\n".join(lines))


def ladder_figure(cfgs):
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.4 * len(cfgs), 2.0), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        df = pd.read_csv(cfg["run_dir"] / "eval" / "overall.csv").set_index("method")
        ms = [m for m in LABEL if m in df.index]
        y = np.arange(len(ms))
        err = np.array([df.loc[ms, "iou"] - df.loc[ms, "iou_lo"], df.loc[ms, "iou_hi"] - df.loc[ms, "iou"]])
        ax.barh(y, df.loc[ms, "iou"], xerr=err, color=["#9aa5b1" if m.startswith("index") else "#2f6db5" for m in ms])
        ax.set_yticks(y, [LABEL[m] for m in ms])
        ax.invert_yaxis()
        ax.set_xlabel("Test IoU")
        ax.set_title(cfg["task"].capitalize())
    fig.savefig(FIGS / "ladder.pdf")
    plt.close(fig)


def band_figure(cfg):
    """Per-band importance from three views: IG share (U-Net), TreeSHAP (LightGBM), occlusion drop (U-Net)."""
    ex = cfg["run_dir"] / "explain"
    attr = pd.read_csv(ex / "band_attribution.csv")
    shap_df = pd.read_csv(ex / "tree_shap.csv")
    occ = pd.read_csv(ex / "occlusion.csv")
    a = attr[(attr.variant == "unet_all") & (attr.state == "trained") & (attr.attribution == "ig")]
    s = shap_df[(shap_df.variant == "lgbm_all") & (shap_df.pixels == "all")]
    o = occ[(occ.method == "unet_all") & occ.group.isin(S2_BANDS)].set_index("group")
    views = {
        "U-Net |IG| share": [a[a.channel.str.endswith(b)].share.sum() for b in S2_BANDS],
        "LightGBM mean |SHAP|": [s[s.channel.str.endswith(b)].mean_abs_shap.sum() for b in S2_BANDS],
        "U-Net IoU drop (occlusion)": [o.loc[b, "iou_drop"] if b in o.index else 0 for b in S2_BANDS],
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.8), sharey=True)
    for ax, (title, v) in zip(axes, views.items()):
        v = np.array(v, float)
        ax.bar(range(len(S2_BANDS)), v / (np.abs(v).max() + 1e-12), color="#2f6db5")
        ax.set_xticks(range(len(S2_BANDS)), S2_BANDS, rotation=90)
        ax.set_title(title)
    axes[0].set_ylabel("relative importance")
    fig.savefig(FIGS / f"bands_{cfg['task']}.pdf")
    plt.close(fig)


def context_figure(cfgs):
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for cfg in cfgs:
        ctx = pd.read_csv(cfg["run_dir"] / "explain" / "context.csv")
        for variant, g in ctx.groupby("variant"):
            ax.plot(g.radius_px * 10, g.mass_share, label=f"{cfg['task']}, {LABEL[variant]}")
    ax.set_xlabel("distance from target pixel (m)")
    ax.set_ylabel("cumulative |IG| share")
    ax.legend(fontsize=6)
    fig.savefig(FIGS / "context.pdf")
    plt.close(fig)


def qualitative_figure(cfg, n=3):
    """Chips where U-Net beats the tuned index by the widest IoU margin."""
    pc = pd.read_csv(cfg["run_dir"] / "eval" / "per_chip.csv")
    pc["iou"] = pc.tp / (pc.tp + pc.fp + pc.fn).clip(lower=1)
    wide = pc.pivot(index="chip", columns="method", values="iou")
    if not {"unet_all", "index_tuned"} <= set(wide.columns):
        return
    top = (wide.unet_all - wide.index_tuned).sort_values(ascending=False).index[:n]
    pred = Predictor(cfg, ["index_tuned", "unet_all"], compute_band_stats(cfg["data_dir"]))
    paths = {p.stem: p for p in chip_paths(cfg["data_dir"], "test")}
    fig, axes = plt.subplots(n, 4, figsize=(7.0, 1.8 * n), squeeze=False)
    for row, cid in zip(axes, top):
        chip, preds = pred.chip(paths[cid])
        img = chip["image"]
        rgb = np.stack([img[3], img[2], img[1]], -1)  # post-fire / composite B04, B03, B02
        rgb = np.clip(rgb / (np.percentile(rgb, 98) + 1e-6), 0, 1)
        for ax, (title, im) in zip(row, [("RGB", rgb), ("Reference", chip["mask"] == 1),
                                         (f"Index ({wide.loc[cid, 'index_tuned']:.2f})", preds["index_tuned"]),
                                         (f"U-Net ({wide.loc[cid, 'unet_all']:.2f})", preds["unet_all"])]):
            ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
            ax.set_title(title)
            ax.axis("off")
    fig.savefig(FIGS / f"examples_{cfg['task']}.pdf")
    plt.close(fig)


def macros(cfgs):
    """\\newcommand macros for scalar results quoted in the text."""
    lines = []
    for cfg in cfgs:
        t = cfg["task"].capitalize()
        df = pd.read_csv(cfg["run_dir"] / "eval" / "overall.csv").set_index("method")
        for m in df.index:
            key = "".join(w.capitalize() for w in m.split("_"))
            lines.append(rf"\newcommand{{\iou{t}{key}}}{{{df.loc[m, 'iou']:.3f}}}")
        sj = cfg["run_dir"] / "explain" / "summary.json"
        if sj.exists():
            for k, v in json.loads(sj.read_text()).items():
                key = "".join(w.capitalize() for w in k.replace("50", "fifty").replace("90", "ninety").split("_"))
                key = "".join(ch for ch in key if ch.isalpha())
                lines.append(rf"\newcommand{{\res{t}{key}}}{{{v:.2f}}}")
    (TABLES / "results_macros.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    cfgs = [load_config(c) for c in parser.parse_args().configs]
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    main_table(cfgs)
    ladder_figure(cfgs)
    context_figure(cfgs)
    for cfg in cfgs:
        for family in ("severity", "landcover", "index_margin", "boundary", "ndvi", "ndvi_pre"):
            gain_table(cfg, family)
        band_figure(cfg)
        qualitative_figure(cfg)
    macros(cfgs)
    print(f"wrote {TABLES} and {FIGS}")
