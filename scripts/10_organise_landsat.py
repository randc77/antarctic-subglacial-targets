#!/usr/bin/env python3
"""
Group flat Landsat band files into one directory per scene.

USGS bulk downloads often land as a flat pile of .TIF and .txt files. The
extraction pipeline expects one directory per scene, because that is how an
untarred product bundle is laid out. This script creates those directories
and moves each file into the right one.

Safe to re-run: files already in place are left alone, and nothing is deleted.

Usage:
    python scripts/organise_landsat.py data/landsat
    python scripts/organise_landsat.py data/landsat --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import defaultdict

# LC08_L2SR_125112_20260223_20260227_02_T2_SR_B1.TIF
#  ^------------------ scene id ------------------^ ^suffix^
SCENE_RE = re.compile(
    r"^(?P<scene>L[COTEM]\d{2}_L2S[PR]_\d{6}_\d{8}_\d{8}_\d{2}_T\d)_(?P<suffix>.+)$"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding the loose Landsat files")
    ap.add_argument("--dry-run", action="store_true", help="show what would move, change nothing")
    a = ap.parse_args()

    if not os.path.isdir(a.root):
        print(f"ERROR: {a.root} is not a directory")
        return 1

    groups: dict[str, list[str]] = defaultdict(list)
    skipped: list[str] = []

    for name in sorted(os.listdir(a.root)):
        full = os.path.join(a.root, name)
        if os.path.isdir(full):
            continue
        m = SCENE_RE.match(name)
        if m:
            groups[m.group("scene")].append(name)
        else:
            skipped.append(name)

    if not groups:
        print(f"No Landsat band files found directly in {a.root}.")
        if skipped:
            print(f"  ({len(skipped)} unrecognised file(s), e.g. {skipped[:3]})")
        return 1

    print(f"Found {len(groups)} scene(s) in {a.root}\n")

    for scene, files in sorted(groups.items()):
        dest = os.path.join(a.root, scene)
        has_st = any("_ST_B10" in f for f in files)
        has_qa = any("_QA_PIXEL" in f for f in files)
        has_mtl = any(f.endswith("_MTL.txt") for f in files)
        sr = sorted(f.split("_")[-1].replace(".TIF", "")
                    for f in files if "_SR_B" in f)

        print(f"  {scene}")
        print(f"      {len(files)} files -> {scene}/")
        print(f"      SR bands: {', '.join(sr) if sr else 'NONE'}")
        print(f"      QA_PIXEL: {'yes' if has_qa else 'NO'}   "
              f"MTL.txt: {'yes' if has_mtl else 'NO'}   "
              f"ST_B10: {'yes' if has_st else 'NO  <-- no thermal band'}")

        if not a.dry_run:
            os.makedirs(dest, exist_ok=True)
            for f in files:
                src = os.path.join(a.root, f)
                dst = os.path.join(dest, f)
                if os.path.exists(dst):
                    continue
                shutil.move(src, dst)
        print()

    if a.dry_run:
        print("DRY RUN - nothing was moved. Re-run without --dry-run to apply.")
    else:
        print("Done. Each scene is now in its own directory.")

    if skipped:
        print(f"\nLeft alone ({len(skipped)} file(s) that are not Landsat bands):")
        for s in skipped[:10]:
            print(f"    {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
