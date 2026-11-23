#!/usr/bin/env python3
"""
Stage 1 extraction: Landsat scenes + outcrop polygons -> training table.

Runs on your machine (needs rasterio + geopandas). Writes a single parquet
file of per-pixel feature vectors, labels, coordinates and spatial-block ids.
That table is small enough to move around, version, and hand to the training
step — which needs nothing but scikit-learn.

Usage
-----
    python scripts/01_extract_stage1.py \
        --config config/lambert_amery.yml \
        --scenes  data/landsat \
        --outcrop data/add/Rock_outcrop_medium_res_polygon.shp \
        --out     data/processed/stage1_lambert_amery.parquet

Add --limit-scenes 2 for a fast first pass before committing to the full set.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aeml import features as F          # noqa: E402
from aeml import labels as L            # noqa: E402
from aeml import landsat as LS          # noqa: E402
from aeml import sampling as S          # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes", required=True, help="directory of untarred Landsat L2 scenes")
    ap.add_argument("--outcrop", required=True, help="ADD rock-outcrop shapefile or gpkg")
    ap.add_argument("--out", required=True, help="output .parquet path")
    ap.add_argument("--limit-scenes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260101)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    region = cfg["region"]
    bbox = (region["xmin"], region["ymin"], region["xmax"], region["ymax"])
    px = float(cfg["extraction"]["pixel_size_m"])
    rng = np.random.default_rng(args.seed)

    scene_dirs = LS.find_scenes(args.scenes)
    if args.limit_scenes:
        scene_dirs = scene_dirs[: args.limit_scenes]
    print(f"Found {len(scene_dirs)} scene(s) under {args.scenes}")

    frames: list[pd.DataFrame] = []

    for i, sd in enumerate(scene_dirs, 1):
        sid = os.path.basename(sd.rstrip("/"))
        print(f"\n[{i}/{len(scene_dirs)}] {sid}")

        mtl = LS.read_mtl(sd)
        warn = LS.geometry_warning(mtl, pixel_size_m=px)
        if warn:
            print(f"    GEOMETRY  {warn}")

        scene = LS.read_scene(sd)
        print(f"    shape {scene.shape}  crs {scene.crs}  valid {scene.valid.mean():.1%}"
              f"  thermal {'yes' if scene.has_thermal else 'NO (L2SR)'}")

        if scene.valid.mean() < cfg["extraction"]["min_valid_fraction"]:
            print("    skipped: too much cloud/fill")
            continue

        feats = F.build_features(scene.bands, texture_size=cfg["extraction"]["texture_window_px"])

        gdf = L.load_outcrop(args.outcrop, scene.crs, bbox=bbox)
        lab = L.rasterize_labels(
            gdf, scene.transform, scene.shape,
            pixel_size_m=px, edge_buffer_m=cfg["extraction"]["edge_buffer_m"],
        )
        counts = S.class_report_counts(lab)
        print(f"    labels: {counts['rock']:,} rock / {counts['nonrock']:,} non / "
              f"{counts['excluded']:,} excluded")

        if counts["rock"] < cfg["extraction"]["min_rock_pixels_per_scene"]:
            print("    skipped: too little outcrop to be worth keeping")
            continue

        X, flat_idx, feat_names = F.stack_to_table(feats, scene.valid)
        if X.shape[0] == 0:
            print("    skipped: no finite rows")
            continue

        y = lab.ravel()[flat_idx]
        xs, ys = LS.pixel_coords(scene.transform, scene.shape)
        px_x, px_y = xs.ravel()[flat_idx], ys.ravel()[flat_idx]

        # Clip to the study region and drop the excluded edge band.
        keep = (
            (px_x >= bbox[0]) & (px_x <= bbox[2])
            & (px_y >= bbox[1]) & (px_y <= bbox[3])
            & (y != S.LABEL_EXCLUDE)
        )
        X, y, px_x, px_y = X[keep], y[keep], px_x[keep], px_y[keep]
        if X.shape[0] == 0:
            print("    skipped: nothing inside the study region")
            continue

        sel = S.stratified_sample(
            y, rng,
            max_rock=cfg["extraction"]["max_rock_px_per_scene"],
            nonrock_per_rock=cfg["extraction"]["nonrock_per_rock"],
        )

        df = pd.DataFrame(X[sel], columns=feat_names)
        df["label"] = y[sel].astype(np.uint8)
        df["x"] = px_x[sel]
        df["y"] = px_y[sel]
        df["scene_id"] = sid
        meta = LS.parse_scene_id(sid)
        df["acq_date"] = meta.get("acq", "")
        # Carried through so geolocation quality can be used as a covariate,
        # a filter, or at minimum reported honestly in the write-up.
        df["tier"] = str(mtl.get("COLLECTION_CATEGORY", "?"))
        df["has_thermal"] = bool(scene.has_thermal)
        df["geom_rmse_m"] = float(mtl.get("GEOMETRIC_RMSE_MODEL", np.nan))
        df["sun_elev"] = float(mtl.get("SUN_ELEVATION", np.nan))
        frames.append(df)
        print(f"    kept {len(df):,} rows "
              f"({int((df.label == S.LABEL_ROCK).sum()):,} rock)")

    if not frames:
        print("\nNo usable scenes. Check cloud cover and that the study-region "
              "bbox actually overlaps your scenes.")
        return 1

    # A thermal and a non-thermal scene produce different columns. Concatenating
    # them would silently fill st_kelvin with NaN for the L2SR rows, and those
    # rows would then be dropped at training time without anyone noticing.
    colsets = {tuple(f.columns) for f in frames}
    if len(colsets) > 1:
        print("\n!! Scenes produced different feature sets (mixed L2SP and L2SR).")
        print("   Keeping only the columns common to all scenes:")
        common = set.intersection(*(set(c) for c in colsets))
        dropped = set.union(*(set(c) for c in colsets)) - common
        print(f"   dropped: {sorted(dropped)}")
        order = [c for c in frames[0].columns if c in common]
        frames = [f[order] for f in frames]

    out = pd.concat(frames, ignore_index=True)
    out["block"] = S.spatial_blocks(
        out["x"].to_numpy(), out["y"].to_numpy(),
        block_km=cfg["extraction"]["block_km"],
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)

    n_rock = int((out.label == S.LABEL_ROCK).sum())
    print(f"\nWrote {args.out}")
    print(f"  rows        {len(out):,}")
    print(f"  rock        {n_rock:,} ({n_rock / len(out):.1%})")
    print(f"  scenes      {out.scene_id.nunique()}")
    print(f"  blocks      {out.block.nunique():,}")
    print(f"  size        {os.path.getsize(args.out) / 1e6:.1f} MB")
    print("\nThis file is the handoff point — training needs only scikit-learn.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
