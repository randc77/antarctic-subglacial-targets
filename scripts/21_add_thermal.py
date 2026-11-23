#!/usr/bin/env python3
"""
Attach basal thermal state to the stage-2 site list.

Reads the Seiner et al. AIS_BaTh_v1 product (netCDF-4), samples it at each
candidate site, and writes an updated CSV. Also writes a small .npz of the
clipped thermal field so the column can be folded into the pipeline proper.

The product is on an 8 km grid against Bedmap3's 500 m, so it describes the
regional setting a site sits in, not the site itself. It is a REPORT column,
never a filter - 36% of the ice sheet is classed "uncertain", and a candidate
in uncertain terrain is not thereby disqualified.

Usage:
    python scripts/add_thermal.py data\\thermal\\AIS_BaTh_v1.nc stage2_sites.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xarray as xr

nc_path = sys.argv[1] if len(sys.argv) > 1 else "AIS_BaTh_v1.nc"
csv_path = sys.argv[2] if len(sys.argv) > 2 else "stage2_sites.csv"

# Lambert-Amery bounds, EPSG:3031
XMIN, XMAX, YMIN, YMAX = 1419000, 2605000, 284000, 1323000

ds = xr.open_dataset(nc_path)

print("=" * 72)
print("AIS_BaTh_v1  -  basal thermal state")
print("=" * 72)
print(f"  dims   {dict(ds.sizes)}")
print("  data variables:")
for n, da in ds.data_vars.items():
    print(f"      {n:<28} {str(da.dims):<22} {str(da.dtype):<9} "
          f"{da.attrs.get('long_name', '')[:40]}")
    for k in ("flag_values", "flag_meanings", "units", "description", "long_name"):
        if k in da.attrs and str(da.attrs[k]).strip():
            print(f"          {k}: {da.attrs[k]}")          # full, not truncated
print("  coordinates:", {k: int(v.size) for k, v in ds.coords.items()})
print("  global attrs:", {k: str(v)[:60] for k, v in list(ds.attrs.items())[:6]})

# --- pick the thermal variable ---
# 'agreement' is the published SYNTHESIS of the two methods - the ternary
# frozen / thawed / uncertain product. 'binaryIsmip' is only the ISMIP6
# model half of it, and 'binary_basalratio_std' only the slip-ratio half.
# Prefer the synthesis; fall back in that order.
PREFER = ["agreement", "binaryIsmip", "binary_basalratio_std"]
two_d = [n for n in ds.data_vars if ds[n].ndim == 2]
var = next((p for p in PREFER if p in two_d), two_d[0])
cands = [var] + [n for n in two_d if n != var]
print(f"\n  using variable: {var}  (the published two-method synthesis)")
print(f"  others present: {cands[1:]}")

da = ds[var]
xname = "x" if "x" in da.dims else list(da.dims)[-1]
yname = "y" if "y" in da.dims else list(da.dims)[0]
gx = ds[xname].values.astype(float)
gy = ds[yname].values.astype(float)
vals = da.values
print(f"  grid spacing: {abs(gx[1]-gx[0]):,.0f} m")
u, c = np.unique(vals[np.isfinite(vals)], return_counts=True)
print("  value counts over the whole ice sheet:")
for a, b in zip(u[:8], c[:8]):
    print(f"      {a:>8.2f}   {b:>9,}  ({100*b/c.sum():5.1f}%)")

# --- sample at the sites ---
sites = pd.read_csv(csv_path)
sx, sy = sites["x_3031"].to_numpy(), sites["y_3031"].to_numpy()
ix = np.clip(np.abs(gx[:, None] - sx[None, :]).argmin(axis=0), 0, len(gx) - 1)
iy = np.clip(np.abs(gy[:, None] - sy[None, :]).argmin(axis=0), 0, len(gy) - 1)
sites["basal_thermal"] = vals[iy, ix]

print(f"\n  sampled at {len(sites)} candidate sites:")
su, sc = np.unique(sites["basal_thermal"].to_numpy()[
    np.isfinite(sites["basal_thermal"].to_numpy())], return_counts=True)
for a, b in zip(su, sc):
    print(f"      {a:>8.2f}   {b:>4,} sites  ({100*b/len(sites):5.1f}%)")
n_nan = int(sites["basal_thermal"].isna().sum())
if n_nan:
    print(f"      (no value)  {n_nan:>4,} sites")

out_csv = csv_path.replace(".csv", "_thermal.csv")
sites.to_csv(out_csv, index=False)

# --- small clipped grid, for folding into the pipeline ---
mx = (gx >= XMIN - 20000) & (gx <= XMAX + 20000)
my = (gy >= YMIN - 20000) & (gy <= YMAX + 20000)
payload = {"x": gx[mx], "y": gy[my],
           "thermal": vals[np.ix_(my, mx)],
           "varname": np.array([var], dtype=object)}
for n in two_d:                       # every layer, so nothing needs re-running
    payload[f"layer_{n}"] = ds[n].values[np.ix_(my, mx)]
    d_ = ds[n].attrs.get("description") or ds[n].attrs.get("long_name") or ""
    payload[f"desc_{n}"] = np.array([str(d_)], dtype=object)
np.savez_compressed("thermal_lambert_amery.npz", **payload)
print(f"\n  wrote {out_csv}")
print("  wrote thermal_lambert_amery.npz  (small - send me this one)")
ds.close()
