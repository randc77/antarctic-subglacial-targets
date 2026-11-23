#!/usr/bin/env python3
"""
Convert the stage-1 parquet to a compressed .npz, and print a summary.

Parquet needs pyarrow, which is not available everywhere. .npz is read by
numpy alone, so this makes the training table portable. Nothing is lost:
the feature matrix, labels, spatial blocks, coordinates and per-scene
metadata all survive.

Usage:
    python scripts/parquet_to_npz.py data/processed/stage1_lambert_amery.parquet
    python scripts/parquet_to_npz.py <in.parquet> --out <out.npz>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

# Columns that are metadata rather than model features.
META_COLS = ("label", "x", "y", "block", "scene_id", "acq_date",
             "tier", "geom_rmse_m", "sun_elev", "has_thermal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or os.path.splitext(a.parquet)[0] + ".npz"

    df = pd.read_parquet(a.parquet)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    print(f"read   {a.parquet}")
    print(f"  rows      {len(df):,}")
    print(f"  features  {len(feat_cols)}  {feat_cols}")
    print(f"  metadata  {[c for c in df.columns if c in META_COLS]}")
    print()

    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.uint8)

    payload = {
        "X": X,
        "y": y,
        "feature_names": np.array(feat_cols, dtype=object),
        "block": df["block"].to_numpy(dtype=np.int64),
        "x": df["x"].to_numpy(dtype=np.float64),
        "y_coord": df["y"].to_numpy(dtype=np.float64),
    }

    # Scene id as codes + lookup, so it round-trips without an object array.
    scenes = pd.Categorical(df["scene_id"])
    payload["scene_code"] = scenes.codes.astype(np.int16)
    payload["scene_names"] = np.array(list(scenes.categories), dtype=object)

    for c in ("tier", "acq_date"):
        if c in df.columns:
            cat = pd.Categorical(df[c].astype(str))
            payload[f"{c}_code"] = cat.codes.astype(np.int16)
            payload[f"{c}_names"] = np.array(list(cat.categories), dtype=object)

    for c in ("geom_rmse_m", "sun_elev"):
        if c in df.columns:
            payload[c] = df[c].to_numpy(dtype=np.float64)
    if "has_thermal" in df.columns:
        payload["has_thermal"] = df["has_thermal"].to_numpy(dtype=bool)

    # --- summary, so the diagnostics are visible before the file moves ---
    n_rock = int((y == 1).sum())
    print("summary")
    print(f"  rock            {n_rock:,} ({n_rock / len(y):.2%} of the sampled table)")
    print(f"  non-rock        {int((y == 0).sum()):,}")
    print(f"  scenes          {len(scenes.categories)}  {list(scenes.categories)}")
    print(f"  spatial blocks  {df['block'].nunique():,}")
    per_block = df.groupby("block")["label"].agg(["size", "sum"])
    print(f"  rows per block     min {per_block['size'].min():,} / "
          f"median {int(per_block['size'].median()):,} / max {per_block['size'].max():,}")
    print(f"  rock per block     min {per_block['sum'].min():,} / "
          f"median {int(per_block['sum'].median()):,} / max {per_block['sum'].max():,}")
    print(f"  blocks with no rock at all: {int((per_block['sum'] == 0).sum()):,}")
    print()
    nan_cols = [c for c in feat_cols if not np.isfinite(df[c].to_numpy()).all()]
    print(f"  non-finite values: {'none' if not nan_cols else nan_cols}")
    print()
    desc = df[feat_cols].describe().T[["min", "mean", "max"]]
    print("  feature ranges:")
    print(desc.to_string(float_format=lambda v: f"{v:11.4f}"))
    print()

    np.savez_compressed(out, **payload)
    print(f"wrote  {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
