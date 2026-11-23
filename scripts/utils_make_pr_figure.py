#!/usr/bin/env python3
"""
Precision-recall figure for the stage-1 README, at TRUE prevalence.

The point of the figure is the gap between what the sampled table reports and
what the classifier would actually deliver on a scene where rock is 0.68% of
pixels. Plotting the sampled curve alone would be the flattering lie.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve

import sys
sys.path.insert(0, "src")
from train_stage1 import (TRUE_PREVALENCE, blocked_split, corrected_precision,
                          rebuild, SEED)

# Reference palette, light surface (validated: all six checks pass)
C_MODEL, C_BASE = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

NPZ = sys.argv[1] if len(sys.argv) > 1 else "data/interim/stage1_lambert_amery.npz"
d = np.load(NPZ, allow_pickle=True)
X, cols = rebuild(d["X"], list(d["feature_names"]))
y, blocks = d["y"].astype(int), d["block"]
keep = np.all(np.isfinite(X), axis=1)
X, y, blocks = X[keep], y[keep], blocks[keep]
sampled_prev = y.mean()

rng = np.random.default_rng(SEED)
tr, te = blocked_split(blocks, y, rng)

m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
m.fit(X[tr], y[tr])
proba = m.predict_proba(X[te])[:, 1]
p, r, _ = precision_recall_curve(y[te], proba)
p_true = np.array([corrected_precision(v, sampled_prev, TRUE_PREVALENCE) for v in p])

# Baseline operating point
ndsi = X[:, cols.index("ndsi")]
bp = (ndsi[te] < 0.30)
b_rec = (bp & (y[te] == 1)).sum() / (y[te] == 1).sum()
b_prec = (bp & (y[te] == 1)).sum() / max(bp.sum(), 1)
b_true = corrected_precision(b_prec, sampled_prev, TRUE_PREVALENCE)

fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(r, p, lw=2, color=C_MODEL, alpha=0.32, zorder=2)
ax.plot(r, p_true, lw=2, color=C_MODEL, zorder=4)

ax.plot([b_rec], [b_true], "o", ms=9, color=C_BASE,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=6)
ax.plot([b_rec], [b_prec], "o", ms=9, color=C_BASE, alpha=0.32,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)

# Direct labels rather than relying on colour alone
ax.annotate("Gradient boosting\nat true prevalence", xy=(0.45, p_true[np.argmin(abs(r - 0.45))]),
            xytext=(0.30, 0.34), color=C_MODEL, fontsize=9.5, ha="left",
            arrowprops=dict(arrowstyle="-", color=C_MODEL, lw=1, alpha=0.6))
ax.annotate("same model, as the sampled\ntable would report it", xy=(0.55, p[np.argmin(abs(r - 0.55))]),
            xytext=(0.30, 0.86), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=INK3, lw=1, alpha=0.6))
ax.annotate(f"NDSI < 0.30 baseline\nrecall {b_rec:.2f}, precision {b_true:.2f}",
            xy=(b_rec, b_true), xytext=(0.56, 0.47),
            color=C_BASE, fontsize=9.5, ha="left",
            arrowprops=dict(arrowstyle="-", color=C_BASE, lw=1, alpha=0.6))

ax.axhline(TRUE_PREVALENCE, color=INK3, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(0.995, TRUE_PREVALENCE + 0.018, f"random guessing ({TRUE_PREVALENCE:.2%} prevalence)",
        ha="right", fontsize=8, color=INK3)

ax.set_xlabel("Recall", fontsize=10, color=INK2)
ax.set_ylabel("Precision", fontsize=10, color=INK2)
ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
ax.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#d6d5d0")
ax.tick_params(colors=INK3, labelsize=9)

ax.legend(handles=[Line2D([], [], color=C_MODEL, lw=2, label="Gradient boosting"),
                   Line2D([], [], marker="o", ls="", color=C_BASE, label="NDSI threshold")],
          loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)

# Title and subtitle in FIGURE coordinates: absolute positions, so the gap
# between them does not move when the axes margins change.
fig.subplots_adjust(top=0.835, left=0.095, right=0.98, bottom=0.115)
fig.text(0.095, 0.935, "A one-line threshold outperforms the learned model",
         fontsize=12.5, color=INK, va="bottom", ha="left")
fig.text(0.095, 0.875, "Rock-exposure classification, Lambert–Amery; spatially held-out blocks",
         fontsize=8.5, color=INK3, va="bottom", ha="left")
fig.savefig("figures/stage1_precision_recall.png", facecolor=SURFACE)
print("wrote stage1_precision_recall.png")
print(f"  baseline  recall {b_rec:.3f}  precision {b_prec:.3f} sampled / {b_true:.3f} true")
