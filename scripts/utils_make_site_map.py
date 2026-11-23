#!/usr/bin/env python3
"""
Stage-2 figure for the README: where the candidate drilling sites are, and the
basal-thermal result that corroborates them.

Left  - bed topography with mapped outcrop, existing ICE-D samples, and the 206
        candidates split by evidence class. The split is the point: a high score
        over a bed that was never sounded is a statement about Bedmap3's
        interpolation, so those sites are drawn small and kept visually separate
        from the 55 that sit over measured bed.
Right - basal thermal state of the candidates against the ice sheet as a whole.
        The ice-sheet distribution is an OUTLINED reference bar, not a second
        series: it is the baseline the candidate bars are read against.

Colours are the first three slots of the reference categorical palette, which
validate all-pairs (scatter) in light mode: worst CVD dE 9.2, normal-vision 24.0.
Terrain is a single-hue grey ramp so it recedes behind the marks.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

NPZ = sys.argv[1] if len(sys.argv) > 1 else "data/interim/stage2_lambert_amery.npz"
SITES = sys.argv[2] if len(sys.argv) > 2 else "results/stage2_sites_final.csv"
ICED = sys.argv[3] if len(sys.argv) > 3 else "data/iced/ICE-D_extract_concs-ages.xlsx"
OUT = sys.argv[4] if len(sys.argv) > 4 else "figures/stage2_site_map.png"
DPI = 165        # ~1785 px wide: 2x a README's display width, no more

# Reference palette, light surface. Slots 1-3 validate all-pairs.
C_FROZEN, C_OTHER, C_INTERP = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
ROCK = "#4a443c"          # mapped outcrop - a neutral, never a series colour
SHELF = "#e8ecef"

# Ice-sheet-wide basal thermal split, Seiner et al. (reproduced: 17.6/36.3/45.9)
ICE_SHEET = {"frozen": 17.6, "uncertain": 36.3, "thawed": 45.9}


def ps_south(lat, lon, phi_c=-71.0):
    """Forward polar stereographic, EPSG:3031. Used only for the graticule."""
    a, e = 6378137.0, 0.08181919084262149
    phi, lam, pc = np.radians(lat), np.radians(lon), np.radians(phi_c)
    t = np.tan(np.pi/4 + phi/2) / (((1 + e*np.sin(phi))/(1 - e*np.sin(phi)))**(e/2))
    tc = np.tan(np.pi/4 + pc/2) / (((1 + e*np.sin(pc))/(1 - e*np.sin(pc)))**(e/2))
    mc = np.cos(pc)/np.sqrt(1 - e**2*np.sin(pc)**2)
    rho = a*mc*t/tc
    return rho*np.sin(lam), rho*np.cos(lam)


def main() -> int:
    d = np.load(NPZ, allow_pickle=True)
    names = [str(s) for s in d["feature_names"]]
    ny, nx = [int(v) for v in d["grid_shape"]]
    gx, gy = d["x"], d["y"]
    bed = d["X"][:, names.index("bed_elevation")].reshape(ny, nx)
    rock = d["is_rock_mask"].reshape(ny, nx)
    mask = d["mask_code"].reshape(ny, nx)

    # Row order is not guaranteed; set imshow origin from the data itself.
    yy = gy.reshape(ny, nx)[:, 0]
    origin = "upper" if yy[0] > yy[-1] else "lower"
    ext = [gx.min()/1e3, gx.max()/1e3, gy.min()/1e3, gy.max()/1e3]

    # Crop to the region that carries data (sites + outcrop), padded, but keep
    # the Amery shelf in frame - it is the orienting landmark.
    s_pre = pd.read_csv(SITES)
    PAD = 90e3
    cx0 = min(s_pre.x_3031.min(), gx[d["is_rock_mask"]].min()) - PAD
    cx1 = max(s_pre.x_3031.max(), gx[d["is_rock_mask"]].max()) + PAD
    cy0 = min(s_pre.y_3031.min(), gy[d["is_rock_mask"]].min()) - PAD
    cy1 = max(s_pre.y_3031.max(), gy[d["is_rock_mask"]].max()) + PAD
    cx0, cx1 = max(cx0, gx.min()), min(cx1, gx.max())
    cy0, cy1 = max(cy0, gy.min()), min(cy1, gy.max())
    crop = [cx0/1e3, cx1/1e3, cy0/1e3, cy1/1e3]
    aspect = (crop[1] - crop[0]) / (crop[3] - crop[2])

    # Size the canvas from the map so neither panel is padded with dead space.
    MAPW, TOP, BOT, GAP, RIGHT, LEFT = 6.05, 1.02, 0.34, 0.86, 0.30, 0.46
    MAPH = MAPW / aspect
    BARW = 3.15
    W = LEFT + MAPW + GAP + BARW + RIGHT
    H = TOP + MAPH + BOT
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([LEFT/W, BOT/H, MAPW/W, MAPH/H])
    bx = fig.add_axes([(LEFT + MAPW + GAP)/W, (BOT + 0.34)/H,
                       BARW/W, (MAPH - 1.26)/H])

    # ---------------- panel A: the map ----------------
    ax.set_facecolor(SURFACE)
    terrain = LinearSegmentedColormap.from_list(
        "bed", ["#f2f1ee", "#d8d6d0", "#b4b1a9", "#8e8a81", "#6b675f"])

    grounded_or_shelf = np.isin(mask, (1.0, 2.0, 3.0))
    bshow = np.where(grounded_or_shelf & np.isfinite(bed), bed, np.nan)
    lo, hi = np.nanpercentile(bshow, [2, 98])
    # Antialiased on the terrain: it is being downsampled by ~2x into the
    # figure, and nearest-neighbour there is just aliasing noise - which also
    # triples the PNG size.
    ax.imshow(bshow, extent=ext, origin=origin, cmap=terrain, vmin=lo, vmax=hi,
              interpolation="antialiased", zorder=1)
    ax.imshow(np.where(mask == 3.0, 1.0, np.nan), extent=ext, origin=origin,
              cmap=LinearSegmentedColormap.from_list("s", [SHELF, SHELF]),
              interpolation="nearest", zorder=2)
    ax.imshow(np.where(rock, 1.0, np.nan), extent=ext, origin=origin,
              cmap=LinearSegmentedColormap.from_list("r", [ROCK, ROCK]),
              interpolation="nearest", zorder=3)

    # graticule, recessive
    gl = dict(color=INK3, lw=0.5, alpha=0.28, zorder=4)
    lbl = dict(fontsize=6.8, color=INK3, zorder=5,
               bbox=dict(boxstyle="square,pad=0.14", fc=SURFACE, ec="none",
                         alpha=0.72))

    def _edge(px_, py_, want):
        """Where a graticule line meets the named frame edge, if it does."""
        inx = (px_ >= crop[0]) & (px_ <= crop[1])
        iny = (py_ >= crop[2]) & (py_ <= crop[3])
        k = np.flatnonzero(inx & iny)
        if k.size == 0:
            return None
        return (px_[k[np.argmin(px_[k])]], py_[k[np.argmin(px_[k])]]) if want == "left" \
            else (px_[k[np.argmax(py_[k])]], py_[k[np.argmax(py_[k])]])

    for lon in (60, 65, 70, 75, 80):
        la = np.linspace(-78.0, -66.0, 400)
        px_, py_ = ps_south(la, np.full_like(la, float(lon)))
        px_, py_ = px_/1e3, py_/1e3
        ax.plot(px_, py_, **gl)
        e = _edge(px_, py_, "top")
        if e and e[1] > crop[3] - 0.035*(crop[3] - crop[2]):
            ax.text(e[0], crop[3] - 0.012*(crop[3] - crop[2]), f"{lon}°E",
                    ha="center", va="top", **lbl)
    for lat in (-76, -74, -72, -70, -68):
        lo_ = np.linspace(55.0, 85.0, 400)
        px_, py_ = ps_south(np.full_like(lo_, float(lat)), lo_)
        px_, py_ = px_/1e3, py_/1e3
        ax.plot(px_, py_, **gl)
        e = _edge(px_, py_, "left")
        if e and e[0] < crop[0] + 0.035*(crop[1] - crop[0]):
            ax.text(crop[0] + 0.010*(crop[1] - crop[0]), e[1], f"{abs(lat)}°S",
                    ha="left", va="center", **lbl)

    # existing ICE-D samples
    try:
        ic = pd.read_excel(ICED)
        sx, sy = ps_south(ic["latitude"].to_numpy(), ic["longitude"].to_numpy())
        inr = ((sx >= gx.min()) & (sx <= gx.max())
               & (sy >= gy.min()) & (sy <= gy.max()))
        ax.scatter(sx[inr]/1e3, sy[inr]/1e3, s=17, facecolors="none",
                   edgecolors=INK2, linewidths=0.7, alpha=0.55, zorder=6)
        n_iced = int(inr.sum())
    except Exception as exc:                       # figure must still draw
        print(f"  (ICE-D layer skipped: {exc})")
        n_iced = 0

    # candidates, by evidence class
    s = pd.read_csv(SITES)
    frozen = s.radar_surveyed & (s.basal_thermal == "frozen")
    other = s.radar_surveyed & ~frozen
    interp = ~s.radar_surveyed
    for m_, c_, sz, a_, lw in ((interp, C_INTERP, 13, 0.80, 0.4),
                               (other, C_OTHER, 34, 0.95, 0.8),
                               (frozen, C_FROZEN, 62, 1.0, 1.0)):
        ax.scatter(s.x_3031[m_]/1e3, s.y_3031[m_]/1e3, s=sz, c=c_, alpha=a_,
                   edgecolors=SURFACE, linewidths=lw, zorder=7 + int(sz),
                   marker="o")

    # the one site the README calls out
    r3 = s[s["rank"] == 3].iloc[0]
    rx, ry = r3.x_3031/1e3, r3.y_3031/1e3
    ax.scatter([rx], [ry], s=250, facecolors="none", edgecolors=INK,
               linewidths=1.1, zorder=80)
    ax.annotate("rank 3", xy=(rx, ry), xytext=(rx - 128, ry - 96),
                fontsize=7.8, color=INK, va="center", ha="right", zorder=81,
                bbox=dict(boxstyle="round,pad=0.30", fc=SURFACE, ec=INK3,
                          lw=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, alpha=0.8,
                                shrinkA=2, shrinkB=9))

    # scale bar
    x0 = crop[0] + 0.115*(crop[1] - crop[0])
    y0 = crop[2] + 0.050*(crop[3] - crop[2])
    ax.plot([x0, x0 + 200], [y0, y0], color=INK, lw=2.2, solid_capstyle="butt",
            zorder=40)
    ax.text(x0 + 100, y0 + 12, "200 km", fontsize=7.5, color=INK2,
            ha="center", va="bottom", zorder=40)

    ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[2], crop[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(INK3); sp.set_linewidth(0.6); sp.set_alpha(0.5)

    handles = [
        Line2D([], [], marker="o", ls="", markersize=7.2, color=C_FROZEN,
               markeredgecolor=SURFACE, markeredgewidth=1.0,
               label=f"measured bed, frozen  ({int(frozen.sum())})"),
        Line2D([], [], marker="o", ls="", markersize=5.6, color=C_OTHER,
               markeredgecolor=SURFACE, markeredgewidth=0.8,
               label=f"measured bed, other thermal class  ({int(other.sum())})"),
        Line2D([], [], marker="o", ls="", markersize=3.8, color=C_INTERP,
               markeredgecolor=SURFACE, markeredgewidth=0.4,
               label=f"interpolated bed — hypotheses  ({int(interp.sum())})"),
        Line2D([], [], marker="o", ls="", markersize=5.2, color="none",
               markerfacecolor="none", markeredgecolor=INK2, markeredgewidth=0.8,
               label=f"existing ICE-D samples  ({n_iced})"),
        Line2D([], [], marker="s", ls="", markersize=6.0, color=ROCK,
               label="mapped rock outcrop"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=7.8,
              labelcolor=INK2, handletextpad=0.6, borderpad=0.7,
              facecolor=SURFACE, edgecolor="none", framealpha=0.90).set_zorder(50)

    # ---------------- panel B: basal thermal state ----------------
    bx.set_facecolor(SURFACE)
    classes = ["frozen", "uncertain", "thawed"]
    usable = s[s.basal_thermal.isin(classes)]
    cand = [100*float((usable.basal_thermal == c).sum())/len(usable) for c in classes]
    ref = [ICE_SHEET[c] for c in classes]
    ypos = np.arange(len(classes))[::-1]
    h = 0.30

    for i, (yp, cv, rv) in enumerate(zip(ypos, cand, ref)):
        bx.barh(yp + 0.19, cv, height=h, color=C_FROZEN, edgecolor=SURFACE,
                linewidth=1.2, zorder=3)
        bx.barh(yp - 0.19, rv, height=h, facecolor="none", edgecolor=INK3,
                linewidth=1.0, linestyle=(0, (3, 2)), zorder=3)
        bx.text(cv + 1.8, yp + 0.19, f"{cv:.0f}%", va="center", ha="left",
                fontsize=8.6, color=INK, fontweight="bold")
        bx.text(rv + 1.6, yp - 0.19, f"{rv:.0f}%", va="center", ha="left",
                fontsize=8.0, color=INK3)

    bx.set_yticks(ypos)
    bx.set_yticklabels(classes, fontsize=9, color=INK2)
    bx.set_xlim(0, 86); bx.set_xticks([0, 25, 50, 75])
    bx.set_xticklabels(["0", "25", "50", "75%"], fontsize=8, color=INK3)
    bx.tick_params(colors=INK3, length=0)
    bx.grid(axis="x", color=INK3, alpha=0.16, lw=0.6, zorder=0)
    bx.set_axisbelow(True)
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.spines["bottom"].set_color(INK3); bx.spines["bottom"].set_alpha(0.4)
    bx.spines["bottom"].set_linewidth(0.6)

    bx.legend(handles=[
        Line2D([], [], marker="s", ls="", markersize=7, color=C_FROZEN,
               label=f"candidate sites (n={len(usable)})"),
        Line2D([], [], ls=(0, (3, 2)), color=INK3, lw=1.0,
               label="ice sheet, for reference")],
        loc="upper right", bbox_to_anchor=(1.0, 1.14), frameon=False,
        fontsize=7.8, labelcolor=INK2, handletextpad=0.7, borderpad=0.0)

    # ---------------- titles, in figure coordinates ----------------
    fig.text(LEFT/W, 1 - 0.30/H, "206 candidate subglacial drilling sites, "
             "Lambert\u2013Amery", fontsize=13.5, color=INK, va="top", ha="left")
    fig.text(LEFT/W, 1 - 0.56/H,
             "Bed shape alone predicts where rock emerges \u2014 ROC-AUC 0.9225, "
             "spatially blocked at 25 km.\nSites scoring \u2265 0.80 under "
             "\u2264 120 m of ice, \u2265 10 km from any existing sample.",
             fontsize=8.3, color=INK3, va="top", ha="left", linespacing=1.5)
    fig.text((LEFT + MAPW + GAP)/W, 1 - 0.56/H,
             "Basal thermal state is independent\ncorroboration, not a filter",
             fontsize=8.3, color=INK2, va="top", ha="left", linespacing=1.5)

    # Not palette-quantised: at 256 colours the terrain greys crowd out the
    # series hues and the orange and aqua drift ~50 and ~85 in RGB off the
    # validated palette. Resolution is trimmed instead (see DPI).
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT}")
    print(f"  measured+frozen {int(frozen.sum())} | measured+other "
          f"{int(other.sum())} | interpolated {int(interp.sum())} | ICE-D {n_iced}")
    print(f"  candidates {[round(v,1) for v in cand]} vs ice sheet {ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
