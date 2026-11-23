#!/usr/bin/env python3
"""
Inspect every downloaded input before running extraction.

Run this FIRST. It is read-only, takes a couple of minutes, and answers the
questions that determine whether the extraction parameters in the config are
right. Each check is independent: a missing dataset prints a warning and the
script carries on.

Usage (all paths optional - omit what you have not downloaded yet):

    python 00_inspect_inputs.py \
        --outcrop  data/add/Rock_outcrop_medium_res_polygon.shp \
        --scenes   data/landsat \
        --bedmap3  data/bedmap3/bedmap3.nc \
        --velocity data/velocity/antarctic_ice_vel_phase_map_v01.nc \
        --config   config/lambert_amery.yml \
        --out      inspection_report.txt

Writes the report to a text file as well as printing it, so you can paste it
back in one piece.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def rule(title: str) -> None:
    say("")
    say("=" * 72)
    say(title)
    say("=" * 72)


def warn(e: Exception, what: str) -> None:
    say(f"  !! could not inspect {what}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- outcrop
def inspect_outcrop(path: str, bbox) -> None:
    rule("1. ADD ROCK OUTCROP  (label source)")
    import geopandas as gpd

    gdf = gpd.read_file(path)
    say(f"  file              {path}")
    say(f"  CRS               {gdf.crs}")
    say(f"  features (total)  {len(gdf):,}")
    say(f"  columns           {list(gdf.columns)}")

    if gdf.crs is None:
        say("  !! NO CRS. The .prj sidecar is missing. labels.load_outcrop will refuse this.")
        return

    if str(gdf.crs).upper() not in ("EPSG:3031",):
        say(f"  .. not EPSG:3031 - will be reprojected on load (fine, just noting it)")
        gdf = gdf.to_crs("EPSG:3031")

    sub = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    say(f"  features in study region   {len(sub):,}")

    if len(sub) == 0:
        say("  !! NOTHING inside the Lambert-Amery bbox. Check the bbox in the config")
        say("     against the shapefile's actual extent:")
        say(f"     shapefile total_bounds = {gdf.total_bounds}")
        return

    # GRIDCODE provenance: 1 = automated from Landsat 8, 0 = manually digitised
    # legacy ADD polygons (mostly south of 82d40'S). Mixing them means training
    # partly on hand-drawn labels while benchmarking an automated method.
    gc_col = next((c for c in sub.columns if c.upper() == "GRIDCODE"), None)
    if gc_col:
        say(f"  {gc_col} distribution in study region:")
        for val, n in sub[gc_col].value_counts().sort_index().items():
            tag = {1: "automated (Landsat 8)", 0: "manual (legacy ADD)"}.get(val, "unknown")
            say(f"      {val} = {n:>7,}   {tag}")
    else:
        say("  .. no GRIDCODE column found - provenance cannot be separated")

    areas = sub.geometry.area  # m^2 in EPSG:3031
    px = 30.0 * 30.0
    say(f"  outcrop area in region     {areas.sum() / 1e6:,.1f} km^2")
    say(f"  polygon area  min / median / max  "
        f"{areas.min():,.0f} / {areas.median():,.0f} / {areas.max():,.0f} m^2")
    say(f"  polygons smaller than 10 Landsat pixels ({10 * px:,.0f} m^2): "
        f"{int((areas < 10 * px).sum()):,} "
        f"({100 * (areas < 10 * px).mean():.1f}%)")
    say("     ^ these are the ones a 60 m edge buffer will erase entirely.")


# ---------------------------------------------------------------- landsat
def inspect_scenes(root: str, aeml_src: str) -> None:
    rule("2. LANDSAT SCENES  (imagery + geolocation quality)")
    sys.path.insert(0, aeml_src)
    from aeml import landsat as LS

    dirs = sorted(d for d in glob.glob(os.path.join(root, "L*_L2S*")) if os.path.isdir(d))
    say(f"  scene directories found: {len(dirs)}")
    if not dirs:
        say(f"  !! none under {root}")
        say("     Scenes must be UNTARRED, each in its own folder, directly under this path.")
        say(f"     What is actually there: {sorted(os.listdir(root))[:10]}")
        return

    say("")
    say(f"  {'scene':<42} {'tier':<5} {'RMSE m':>7} {'GCPs':>5} {'sun':>5}  ST_B10")
    say(f"  {'-' * 42} {'-' * 5} {'-' * 7} {'-' * 5} {'-' * 5}  {'-' * 6}")
    for d in dirs:
        sid = os.path.basename(d)
        try:
            m = LS.read_mtl(d)
            tier = str(m.get("COLLECTION_CATEGORY", "?"))
            rmse = m.get("GEOMETRIC_RMSE_MODEL")
            gcps = m.get("GROUND_CONTROL_POINTS_MODEL", "-")
            sun = m.get("SUN_ELEVATION")
            has_st = "yes" if glob.glob(os.path.join(d, "*_ST_B10.TIF")) else "NO"
            say(f"  {sid:<42} {tier:<5} "
                f"{(f'{rmse:.1f}' if rmse is not None else 'none'):>7} "
                f"{str(gcps):>5} "
                f"{(f'{sun:.1f}' if sun is not None else '-'):>5}  {has_st}")
            w = LS.geometry_warning(m)
            if w:
                say(f"      -> {w}")
        except Exception as e:  # noqa: BLE001
            say(f"  {sid:<42} !! {type(e).__name__}: {e}")

    say("")
    say("  Rows with ST_B10 = NO are L2SR products: no thermal band, and read_scene")
    say("  will refuse them. Re-download those as L2SP.")


# ---------------------------------------------------------------- netcdf
def inspect_netcdf(path: str, label: str) -> None:
    rule(f"{label}")
    import xarray as xr

    ds = xr.open_dataset(path, decode_coords="all")
    say(f"  file   {path}")
    say(f"  dims   {dict(ds.sizes)}")
    say("")
    say("  variables:")
    for name, da in ds.data_vars.items():
        units = da.attrs.get("units", "")
        ln = da.attrs.get("long_name", da.attrs.get("standard_name", ""))
        say(f"      {name:<28} {str(da.dims):<24} {str(da.dtype):<8} {units:<12} {ln[:40]}")
    say("")
    say("  coordinates:")
    for name, da in ds.coords.items():
        try:
            rng = f"{float(da.min()):,.0f} .. {float(da.max()):,.0f}"
        except Exception:  # noqa: BLE001
            rng = "(non-numeric)"
        say(f"      {name:<28} {da.size:>8,}   {rng}")
    say("")
    say("  global attributes (CRS / datum are what matter):")
    for k, v in ds.attrs.items():
        vs = str(v).replace("\n", " ")
        say(f"      {k:<28} {vs[:90]}")
    ds.close()


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outcrop")
    ap.add_argument("--scenes")
    ap.add_argument("--bedmap3")
    ap.add_argument("--velocity")
    ap.add_argument("--config", default="config/lambert_amery.yml")
    ap.add_argument("--src", default="src", help="path to the aeml package parent dir")
    ap.add_argument("--out", default="inspection_report.txt")
    a = ap.parse_args()

    say("INPUT INSPECTION REPORT")
    say(f"generated by 00_inspect_inputs.py")

    bbox = (1419000, 284000, 2605000, 1323000)  # fallback
    try:
        import yaml
        with open(a.config) as fh:
            cfg = yaml.safe_load(fh)
        r = cfg["region"]
        bbox = (r["xmin"], r["ymin"], r["xmax"], r["ymax"])
        say(f"study region from {a.config}: {bbox}")
    except Exception as e:  # noqa: BLE001
        say(f"!! could not read config ({e}); using built-in Lambert-Amery bbox {bbox}")

    for path, fn, what in (
        (a.outcrop, lambda p: inspect_outcrop(p, bbox), "outcrop shapefile"),
        (a.scenes, lambda p: inspect_scenes(p, a.src), "Landsat scenes"),
        (a.bedmap3, lambda p: inspect_netcdf(p, "3. BEDMAP3"), "bedmap3.nc"),
        (a.velocity, lambda p: inspect_netcdf(p, "4. ICE VELOCITY (NSIDC-0754)"), "velocity netCDF"),
    ):
        if not path:
            continue
        if not os.path.exists(path):
            rule(f"SKIPPED - path does not exist: {path}")
            continue
        try:
            fn(path)
        except Exception as e:  # noqa: BLE001
            warn(e, what)
            LINES.append(traceback.format_exc())

    rule("END OF REPORT")
    with open(a.out, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print(f"\nReport written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
