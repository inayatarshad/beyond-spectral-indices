"""Cross-validation ladder figure (paper/figures/cv_ladder.pdf) from the per-fold test IoUs.

    python scripts/make_cv_figure.py paper/figures
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
out = sys.argv[1]
M = ["Lit.\ndNBR","Otsu\ndNBR","Tuned\ndNBR","GBDT\nidx","GBDT\nall","U-Net\nidx","U-Net\nall"]
f0 = [.3969,.1604,.5182,.5098,.5403,.7102,.7388]
f1 = [.0743,.0453,.1510,.2671,.2029,.2947,.3024]
lo0=[.3309,.1289,.4510,.4539,.4799,.6341,.6573]; hi0=[.4612,.1930,.5797,.5548,.5924,.7755,.8094]
lo1=[.0488,.0285,.1002,.1971,.1249,.1835,.1887]; hi1=[.1041,.0646,.2068,.3410,.2846,.4214,.4334]
plt.rcParams.update({"font.size":7.5,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(figsize=(3.45,1.9))
x = np.arange(len(M)); w=.36
c0, c1 = "#3b6ea8", "#e08a3c"
e0 = [np.subtract(f0,lo0), np.subtract(hi0,f0)]; e1=[np.subtract(f1,lo1), np.subtract(hi1,f1)]
ax.bar(x-w/2, f0, w, color=c0, yerr=e0, capsize=1.5, error_kw=dict(lw=.6), label="Test fold 0")
ax.bar(x+w/2, f1, w, color=c1, yerr=e1, capsize=1.5, error_kw=dict(lw=.6), label="Test fold 1")
mean = (np.array(f0)+np.array(f1))/2
ax.scatter(x, mean, marker="D", s=10, color="k", zorder=5, label="Mean")
ax.axhline(mean[2], ls=":", lw=.7, color="grey")
ax.axvspan(-.5, 2.5, color="0.93", zorder=0)
ax.set_xticks(x, M); ax.set_ylabel("Test IoU"); ax.set_ylim(0, .85); ax.set_xlim(-.5, 6.5)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.5, handlelength=1)
fig.tight_layout(pad=.2)
fig.savefig(f"{out}/cv_ladder.pdf"); fig.savefig(f"{out}/cv_ladder.png", dpi=200)
