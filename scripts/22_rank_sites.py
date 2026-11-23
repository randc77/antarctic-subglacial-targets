#!/usr/bin/env python3
"""
Stage 2 final: rank candidate drilling SITES, not cells.

Model uses bed SHAPE only - relief at 10 and 50 km, and bed slope. Chosen by
ablation rather than by permutation importance:

    all five bed features    AUC 0.969  AP 0.769
    without dist_to_water    AUC 0.937  AP 0.647
    relief + slope only      AUC 0.933  AP 0.646   <- this
    dist_to_water alone      AUC 0.819  AP 0.249
    bed_elevation alone      AUC 0.769  AP 0.276

Dropping bed_elevation and dist_to_water costs almost nothing, and buys a
model that cannot simply have memorised where in the region the Prince
Charles Mountains are. Permutation importance had ranked dist_to_water first;
the ablation shows that was redundancy, not signal.

Output is SITES. An earlier cell-level ranking put the same 2 km massif in
the top 50 seven times over, which is not a target list.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

NPZ  = sys.argv[1] if len(sys.argv) > 1 else "data/interim/stage2_lambert_amery.npz"
ICED = sys.argv[2] if len(sys.argv) > 2 else "data/iced/ICE-D_extract_concs-ages.xlsx"
OUT  = sys.argv[3] if len(sys.argv) > 3 else "results/stage2_sites.csv"
SEED = 20260101
FEATURES = ["bed_relief_10km", "bed_relief_50km", "bed_slope_deg"]
MAX_THICK, MIN_DIST_ICED, BLOCK_KM, NEG_PER_POS = 120.0, 10.0, 25.0, 10
SCORE_MIN = 0.80


def ps_south(lat, lon, phi_c=-71.0):
    a, e = 6378137.0, 0.08181919084262149
    phi, lam, pc = np.radians(lat), np.radians(lon), np.radians(phi_c)
    t = np.tan(np.pi/4 + phi/2) / (((1 + e*np.sin(phi))/(1 - e*np.sin(phi)))**(e/2))
    tc = np.tan(np.pi/4 + pc/2) / (((1 + e*np.sin(pc))/(1 - e*np.sin(pc)))**(e/2))
    mc = np.cos(pc)/np.sqrt(1 - e**2*np.sin(pc)**2)
    rho = a*mc*t/tc
    return rho*np.sin(lam), rho*np.cos(lam)


def xy_to_lonlat(x, y, phi_c=-71.0):
    """Inverse of the above, by series expansion - good to <1 m here."""
    a, e = 6378137.0, 0.08181919084262149
    pc = np.radians(phi_c)
    tc = np.tan(np.pi/4 + pc/2) / (((1 + e*np.sin(pc))/(1 - e*np.sin(pc)))**(e/2))
    mc = np.cos(pc)/np.sqrt(1 - e**2*np.sin(pc)**2)
    rho = np.hypot(x, y)
    t = rho * tc / (a * mc)
    chi = 2*np.arctan(t) - np.pi/2
    phi = chi + (e**2/2 + 5*e**4/24)*np.sin(2*chi) + (7*e**4/48)*np.sin(4*chi)
    return np.degrees(np.arctan2(x, y)), np.degrees(phi)


def blocked_split(blocks, y, rng, test_frac=0.30):
    uniq = np.unique(blocks)
    has = np.array([y[blocks == b].any() for b in uniq])
    test = set()
    for s in (uniq[has], uniq[~has]):
        s = s.copy(); rng.shuffle(s)
        test.update(s[: max(1, int(round(test_frac*len(s))))].tolist())
    return ~np.isin(blocks, list(test)), np.isin(blocks, list(test))


def main() -> int:
    rng = np.random.default_rng(SEED)
    d = np.load(NPZ, allow_pickle=True)
    names = [str(s) for s in d["feature_names"]]
    X = d["X"]; label = d["label"].astype(int)
    grounded = d["grounded"]; rock = d["is_rock_mask"]
    gx, gy = d["x"], d["y"]
    ny, nx = [int(v) for v in d["grid_shape"]]; px = float(d["px_m"])
    thick = X[:, names.index("ice_thickness")]
    bed_unc = d["report_bed_uncertainty"]
    survey = d["report_thickness_survey_count"]
    relief10 = X[:, names.index("bed_relief_10km")]

    Xb = X[:, [names.index(c) for c in FEATURES]]
    fin = np.all(np.isfinite(Xb), axis=1) & np.isfinite(thick)

    print("=" * 76)
    print("STAGE 2 - candidate subglacial drilling SITES, Lambert-Amery")
    print("=" * 76)
    print(f"  predictors   {FEATURES}  (bed shape only)")

    # ---- ICE-D ----
    ic = pd.read_excel(ICED)
    sx, sy = ps_south(ic["latitude"].to_numpy(), ic["longitude"].to_numpy())
    inreg = (sx >= gx.min()) & (sx <= gx.max()) & (sy >= gy.min()) & (sy <= gy.max())
    site_g = np.zeros((ny, nx), bool)
    yy = gy.reshape(ny, nx)[:, 0]
    col = np.clip(((sx[inreg] - gx.min())/px).round().astype(int), 0, nx-1)
    row = np.clip(np.abs(yy[:, None] - sy[inreg][None, :]).argmin(axis=0), 0, ny-1)
    site_g[row, col] = True
    dist_iced = (ndimage.distance_transform_edt(~site_g, sampling=px)/1000.0).ravel()
    print(f"  ICE-D        {int(inreg.sum())} samples at "
          f"{ic.loc[inreg,'iced_site_name'].nunique()} sites in region")

    # ---- train ----
    pos = grounded & (label == 1) & fin
    negp = grounded & (label == 0) & ~rock & fin
    ni = rng.choice(np.flatnonzero(negp),
                    size=min(int(negp.sum()), NEG_PER_POS*int(pos.sum())), replace=False)
    idx = np.concatenate([np.flatnonzero(pos), ni]); rng.shuffle(idx)
    Xt, yt = Xb[idx], (label[idx] == 1).astype(int)
    b = BLOCK_KM*1000
    bxx = np.floor(gx[idx]/b).astype(int); byy = np.floor(gy[idx]/b).astype(int)
    bl = (bxx - bxx.min())*(byy.max() - byy.min() + 1) + (byy - byy.min())
    tr, te = blocked_split(bl, yt, rng)
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                       random_state=SEED).fit(Xt[tr], yt[tr])
    p = m.predict_proba(Xt[te])[:, 1]
    print(f"\n  VALIDATION   spatially blocked, {BLOCK_KM:.0f} km blocks")
    print(f"               ROC-AUC {roc_auc_score(yt[te], p):.4f}   "
          f"avg precision {average_precision_score(yt[te], p):.4f}")
    print("               Bed shape alone predicts where rock emerges, with no")
    print("               knowledge of the ice above it and no geographic proxy.")

    # ---- candidates, with the physical checks applied BEFORE ranking ----
    cand = (grounded & (label == 0) & ~rock & fin
            & (thick > 0) & (thick <= MAX_THICK)
            & (dist_iced >= MIN_DIST_ICED)
            & (relief10 > 0)                      # a peak, not a hollow
            & np.isfinite(bed_unc) & (bed_unc < thick))   # bed better known than ice is thick
    print(f"\n  CANDIDATE CELLS after filters: {int(cand.sum()):,}")
    print(f"    grounded, 0 < ice <= {MAX_THICK:.0f} m, not outcrop,")
    print(f"    >= {MIN_DIST_ICED:.0f} km from ICE-D, positive bed relief,")
    print(f"    bed uncertainty < ice thickness")

    score = np.zeros(len(thick), np.float32)
    ci = np.flatnonzero(cand)
    score[ci] = m.predict_proba(Xb[ci])[:, 1]

    # ---- cluster cells into sites ----
    keep = cand & (score >= SCORE_MIN)
    lab, nlab = ndimage.label(keep.reshape(ny, nx), structure=np.ones((3, 3)))
    print(f"  cells scoring >= {SCORE_MIN}: {int(keep.sum()):,}  ->  {nlab:,} distinct sites")

    flat = lab.ravel()
    rows = []
    for k in range(1, nlab + 1):
        s = np.flatnonzero(flat == k)
        best = s[np.argmax(score[s])]
        has_radar = bool(np.nansum(survey[s]) > 0)
        rows.append(dict(
            site_id=k,
            score_max=float(score[s].max()),
            n_cells=len(s),
            area_km2=len(s)*(px/1000)**2,
            x_3031=float(gx[best]), y_3031=float(gy[best]),
            ice_thickness_m=float(thick[best]),
            thinnest_ice_m=float(np.nanmin(thick[s])),
            bed_relief_10km_m=float(relief10[s].max()),
            bed_uncertainty_m=float(np.nanmin(bed_unc[s])),
            dist_to_iced_km=float(dist_iced[s].min()),
            radar_surveyed=has_radar,
            survey_count=float(np.nansum(survey[s])),
        ))
    sites = pd.DataFrame(rows)
    lon, lat = xy_to_lonlat(sites.x_3031.to_numpy(), sites.y_3031.to_numpy())
    sites["lon"], sites["lat"] = lon, lat
    sites = sites.sort_values(["radar_surveyed", "score_max", "bed_relief_10km_m"],
                              ascending=[False, False, False]).reset_index(drop=True)
    sites.insert(0, "rank", np.arange(1, len(sites) + 1))

    nrad = int(sites.radar_surveyed.sum())
    print(f"\n  sites with at least one radar sounding: {nrad:,} of {len(sites):,}")
    print("  Sites are ranked RADAR-SURVEYED FIRST. A high score over a bed that")
    print("  was never sounded is a statement about an interpolation, not a peak.")

    cols = ["rank", "lat", "lon", "area_km2", "thinnest_ice_m", "bed_relief_10km_m",
            "bed_uncertainty_m", "dist_to_iced_km", "survey_count", "score_max"]
    print("\n" + "-"*76)
    print("TOP SITES WITH MEASURED BED")
    print("-"*76)
    with pd.option_context("display.width", 220, "display.max_columns", 24):
        print(sites[sites.radar_surveyed][cols].head(15).to_string(
            index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\n" + "-"*76)
    print("TOP SITES ON INTERPOLATED BED  (treat as hypotheses, not targets)")
    print("-"*76)
    with pd.option_context("display.width", 220, "display.max_columns", 24):
        print(sites[~sites.radar_surveyed][cols].head(8).to_string(
            index=False, float_format=lambda v: f"{v:,.2f}"))

    sites.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(sites):,} sites)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
