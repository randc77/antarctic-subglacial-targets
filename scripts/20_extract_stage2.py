#!/usr/bin/env python3
"""
Stage 2 extraction: Bedmap3 + velocity + outcrop -> candidate table.

Builds the 500 m predictor stack for subglacial drilling-target identification.
Positives are grid cells containing exposed rock; the model learns what an
outcrop setting looks like in bed-and-ice terms, and stage 3 inverts it to
score ice-covered cells.

Design notes
------------
Two HARD filters, applied at ranking time, not here:
    ice thickness <= 120 m      (realistic bedrock-recovery depth)
    >= 10 km from an ICE-D site (a site that close adds little new information)

Everything else is REPORTED per candidate rather than thresholded, because the
thresholds would be arbitrary and the judgement is the user's:
    basal thermal state, bed uncertainty, thickness uncertainty,
    thickness_survey_count, surface speed, bed relief.

Surface velocity is deliberately NOT a filter. What erases a cosmogenic signal
in subglacial bedrock is a warm-based, sliding bed; cold-based ice can preserve
a surface for millions of years while still deforming internally, and slow ice
can be warm-based. Velocity is kept as a predictor because it describes the
setting, not as a gate on drillability.

Usage:
    python scripts/03_extract_stage2.py \
        --config   config/lambert_amery.yml \
        --bedmap3  data/bedmap3/bedmap3.nc \
        --velocity data/velocity/antarctic_ice_vel_phase_map_v01.nc \
        --outcrop  data/add/Landsat_8_Derived_Outcrop_Dataset_2016.shp \
        --out      data/processed/stage2_lambert_amery.npz

Optional, added as columns when supplied:
    --thermal  data/thermal/<Seiner et al. basal thermal state>.nc
    --iced     data/iced/iced_antarctica.csv   (needs x,y or lat,lon columns)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Bedmap3 ice mask codes, read from the file's own flag_meanings:
#   "grounded_ice transiently_grounded_ice_shelf floating_ice_shelf rock"
# There is no ocean code; open sea falls outside the mask (non-finite).
#
# Getting these wrong is not a subtle error. An earlier guess had 3=rock and
# 4=ocean, which would have labelled the floating Amery Ice Shelf as exposed
# bedrock and fed it to the model as a positive.
MASK_GROUNDED = 1          # grounded ice - the candidate pool
MASK_TRANSIENT = 2         # transiently grounded ice shelf
MASK_SHELF = 3             # floating ice shelf
MASK_ROCK = 4              # exposed rock

RELIEF_RADII_KM = (10.0, 50.0)


def say(s=""):
    print(s, flush=True)


# ----------------------------------------------------------------- helpers
def local_relief(bed, px_m, radius_km):
    """
    How far a cell's bed stands above its surroundings.

    This is the term that actually finds buried peaks: absolute bed elevation
    says little, but a cell 400 m above the mean bed within 10 km is a
    mountain whether or not it is currently exposed.
    """
    from scipy import ndimage
    size = max(3, int(round(2 * radius_km * 1000 / px_m)) | 1)
    finite = np.isfinite(bed)
    filled = np.where(finite, bed, 0.0)
    w = finite.astype(np.float64)
    s = ndimage.uniform_filter(filled, size=size, mode="nearest")
    n = ndimage.uniform_filter(w, size=size, mode="nearest")
    out = np.full(bed.shape, np.nan, np.float32)
    ok = n > 1e-6
    out[ok] = (bed[ok] - s[ok] / n[ok]).astype(np.float32)
    out[~finite] = np.nan
    return out


def slope_deg(z, px_m):
    """Surface gradient magnitude in degrees."""
    gy, gx = np.gradient(z.astype(np.float64), px_m, px_m)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def distance_km(mask_true, px_m):
    """Euclidean distance (km) from every cell to the nearest True cell."""
    from scipy import ndimage
    if not mask_true.any():
        return np.full(mask_true.shape, np.nan, np.float32)
    return (ndimage.distance_transform_edt(~mask_true, sampling=px_m)
            / 1000.0).astype(np.float32)


def resample_to(src_x, src_y, src_v, dst_x, dst_y):
    """
    Nearest-neighbour resample between two regular EPSG:3031 grids.

    Both Bedmap3 and the MEaSUREs velocity map are polar stereographic on the
    same datum, so this is index arithmetic rather than a reprojection.
    """
    xi = np.clip(np.searchsorted(src_x, dst_x), 0, len(src_x) - 1)
    yi = np.clip(np.searchsorted(src_y, dst_y), 0, len(src_y) - 1)
    return src_v[np.ix_(yi, xi)]


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--bedmap3", required=True)
    ap.add_argument("--outcrop", required=True)
    ap.add_argument("--velocity")
    ap.add_argument("--thermal")
    ap.add_argument("--iced")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import xarray as xr
    import yaml

    with open(a.config) as fh:
        cfg = yaml.safe_load(fh)
    reg = cfg["region"]
    bbox = (reg["xmin"], reg["ymin"], reg["xmax"], reg["ymax"])

    # ---------------- Bedmap3 ----------------
    say("=" * 70)
    say("BEDMAP3")
    say("=" * 70)
    ds = xr.open_dataset(a.bedmap3, decode_coords="all")
    sub = ds.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))
    if sub.sizes["x"] == 0 or sub.sizes["y"] == 0:   # y may descend
        sub = ds.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))
    gx = sub["x"].values.astype(np.float64)
    gy = sub["y"].values.astype(np.float64)
    px = float(abs(gx[1] - gx[0]))
    say(f"  grid        {sub.sizes['y']} x {sub.sizes['x']} at {px:.0f} m")
    say(f"  x range     {gx.min():,.0f} .. {gx.max():,.0f}")
    say(f"  y range     {gy.min():,.0f} .. {gy.max():,.0f}")

    def g(name):
        return sub[name].values.astype(np.float32) if name in sub else None

    bed = g("bed_topography")
    thick = g("ice_thickness")
    surf = g("surface_topography")
    mask = g("mask")
    bed_unc = g("bed_uncertainty")
    thk_unc = g("thickness_uncertainty")
    survey = g("thickness_survey_count")
    if bed is None or thick is None or mask is None:
        say("  !! missing a required Bedmap3 variable; check names in the file")
        return 1

    codes, counts = np.unique(mask[np.isfinite(mask)], return_counts=True)
    say("  mask codes found (verify these against the file's own metadata):")
    for c, n in zip(codes, counts):
        say(f"      {c:>6.0f}  {n:>10,} cells  ({100*n/mask.size:5.2f}%)")

    grounded = mask == MASK_GROUNDED
    rock_mask = mask == MASK_ROCK
    # Transiently grounded ice sits near flotation: grounded only part of the
    # time, so its bed is marginal and very likely wet. Not a drilling target,
    # and grouped with floating ice rather than with grounded.
    floating = (mask == MASK_SHELF) | (mask == MASK_TRANSIENT)
    water = floating | ~np.isfinite(mask)      # no ocean code; sea is non-finite
    say(f"  grounded ice            {grounded.sum():>10,}")
    say(f"  floating + transient    {floating.sum():>10,}   <- excluded from candidates")
    say(f"  exposed rock            {rock_mask.sum():>10,}")
    say(f"  outside mask (ocean)    {int((~np.isfinite(mask)).sum()):>10,}")
    say("  SANITY CHECK: the Amery Ice Shelf is a large part of this region, so")
    say("  'floating + transient' should be a substantial count. If it is near")
    say("  zero, the mask codes are wrong and nothing downstream can be trusted.")

    # ---------------- derived terrain ----------------
    say("\n" + "=" * 70)
    say("DERIVED TERRAIN")
    say("=" * 70)
    feats, names = [], []

    def add(name, arr):
        feats.append(arr.astype(np.float32)); names.append(name)
        f = np.isfinite(arr)
        say(f"  {name:22s} {np.nanmin(arr):10.1f} .. {np.nanmax(arr):10.1f}  "
            f"({100*f.mean():5.1f}% finite)")

    add("bed_elevation", bed)
    add("ice_thickness", thick)
    if surf is not None:
        add("surface_elevation", surf)
        add("surface_slope_deg", slope_deg(surf, px))
    add("bed_slope_deg", slope_deg(bed, px))
    for rkm in RELIEF_RADII_KM:
        add(f"bed_relief_{int(rkm)}km", local_relief(bed, px, rkm))
    add("dist_to_water_km", distance_km(water, px))

    # ---------------- velocity ----------------
    if a.velocity and os.path.exists(a.velocity):
        say("\n" + "=" * 70)
        say("VELOCITY  (a predictor, not a filter - see module docstring)")
        say("=" * 70)
        v = xr.open_dataset(a.velocity, decode_coords="all")
        vx_all, vy_all = v["VX"].values, v["VY"].values
        vgx, vgy = v["x"].values.astype(np.float64), v["y"].values.astype(np.float64)
        ox = np.argsort(vgx); oy = np.argsort(vgy)
        spd_all = np.hypot(vx_all, vy_all)[np.ix_(oy, ox)]
        spd = resample_to(vgx[ox], vgy[oy], spd_all, gx, gy)
        if gy[0] > gy[-1]:
            pass  # resample_to already indexed by value, orientation preserved
        add("surface_speed_m_yr", spd)
        if "ERRX" in v and "ERRY" in v:
            err_all = np.hypot(v["ERRX"].values, v["ERRY"].values)[np.ix_(oy, ox)]
            add("speed_error_m_yr", resample_to(vgx[ox], vgy[oy], err_all, gx, gy))
        v.close()
    else:
        say("\n  (no velocity supplied - skipping)")

    # ---------------- labels ----------------
    say("\n" + "=" * 70)
    say("LABELS  (outcrop polygons -> 500 m cells)")
    say("=" * 70)
    import geopandas as gpd
    from rasterio import features as rfeatures
    from rasterio.transform import from_origin

    gdf = gpd.read_file(a.outcrop).to_crs("EPSG:3031")
    gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    say(f"  polygons in region: {len(gdf):,}")

    top = gy.max() + px / 2
    tf = from_origin(gx.min() - px / 2, top, px, px)
    shp = (len(gy), len(gx))
    burn = rfeatures.rasterize(
        [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=shp, transform=tf, fill=0, dtype=np.uint8, all_touched=True)
    if gy[0] < gy[-1]:
        burn = burn[::-1]              # rasterio writes north-up
    label = burn.astype(np.uint8)
    say(f"  cells containing outcrop: {int(label.sum()):,} "
        f"({100*label.mean():.3f}% of region)")

    # ---------------- optional layers ----------------
    thermal = None
    if a.thermal and os.path.exists(a.thermal):
        say("\n  basal thermal state: loading")
        t = xr.open_dataset(a.thermal, decode_coords="all")
        tv = [k for k in t.data_vars][0]
        say(f"      using variable '{tv}' of {list(t.data_vars)}")
        tgx, tgy = t["x"].values.astype(np.float64), t["y"].values.astype(np.float64)
        tox, toy = np.argsort(tgx), np.argsort(tgy)
        thermal = resample_to(tgx[tox], tgy[toy],
                              t[tv].values[np.ix_(toy, tox)], gx, gy)
        say("      8 km grid resampled to 500 m: regional context only, not a "
            "per-cell discriminator")
        t.close()
    else:
        say("\n  (no basal thermal state supplied - column omitted)")

    dist_iced = None
    if a.iced and os.path.exists(a.iced):
        say("\n  ICE-D sites: loading")
        import pandas as pd
        ic = pd.read_csv(a.iced)
        cols = {c.lower(): c for c in ic.columns}
        if "x" in cols and "y" in cols:
            sx, sy = ic[cols["x"]].to_numpy(), ic[cols["y"]].to_numpy()
        elif "lat" in cols and "lon" in cols:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
            sx, sy = tr.transform(ic[cols["lon"]].to_numpy(), ic[cols["lat"]].to_numpy())
        else:
            say(f"      !! need x/y or lat/lon columns; found {list(ic.columns)[:8]}")
            sx = sy = None
        if sx is not None:
            site = np.zeros(shp, dtype=bool)
            ix = np.searchsorted(np.sort(gx), sx)
            iy = np.searchsorted(np.sort(gy), sy)
            ok = (ix > 0) & (ix < len(gx)) & (iy > 0) & (iy < len(gy))
            site[iy[ok], ix[ok]] = True
            say(f"      {int(site.sum()):,} of {len(ic):,} sites fall inside the region")
            dist_iced = distance_km(site, px)
    else:
        say("\n  (no ICE-D export supplied - distance column omitted)")

    # ---------------- assemble ----------------
    say("\n" + "=" * 70)
    say("OUTPUT")
    say("=" * 70)
    X = np.stack(feats, axis=-1).reshape(-1, len(feats))
    yy, xx = np.meshgrid(gy, gx, indexing="ij")

    payload = dict(
        X=X.astype(np.float32),
        feature_names=np.array(names, dtype=object),
        label=label.ravel(),
        x=xx.ravel().astype(np.float64),
        y=yy.ravel().astype(np.float64),
        grounded=grounded.ravel(),
        is_rock_mask=rock_mask.ravel(),
        mask_code=mask.ravel().astype(np.float32),
        px_m=np.float64(px),
        grid_shape=np.array(shp, dtype=np.int64),
    )
    for nm, arr in (("bed_uncertainty", bed_unc),
                    ("thickness_uncertainty", thk_unc),
                    ("thickness_survey_count", survey),
                    ("basal_thermal", thermal),
                    ("dist_to_iced_km", dist_iced)):
        if arr is not None:
            payload[f"report_{nm}"] = arr.ravel().astype(np.float32)
            say(f"  report column: {nm}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **payload)

    thin = grounded & np.isfinite(thick) & (thick <= 120) & (label == 0)
    say(f"\n  cells                     {X.shape[0]:,}")
    say(f"  features                  {len(names)}  {names}")
    say(f"  outcrop cells (positives) {int(label.sum()):,}")
    say(f"  grounded, ice <= 120 m,")
    say(f"    not already outcrop     {int(thin.sum()):,}  <- the candidate pool")
    say(f"  wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
