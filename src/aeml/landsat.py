"""
Landsat 8/9 Collection 2 Level-2 scene reading.

Requires rasterio. Isolated here so that `features.py` and `sampling.py`
stay importable (and testable) in environments without GDAL.

Expected on-disk layout — one directory per untarred scene:

    LC08_L2SP_127110_20200115_20200824_02_T1/
        LC08_L2SP_127110_20200115_20200824_02_T1_SR_B1.TIF
        ... SR_B2 .. SR_B7 ...
        LC08_L2SP_127110_20200115_20200824_02_T1_ST_B10.TIF
        LC08_L2SP_127110_20200115_20200824_02_T1_QA_PIXEL.TIF

Antarctic projection note
-------------------------
Landsat Collection 2 scenes south of ~63 deg S are delivered in WGS 84 /
Antarctic Polar Stereographic (EPSG:3031) rather than UTM. That is the same
CRS as Bedmap3 and REMA, so for this study region no reprojection is needed
between stage 1 and stage 2. Verify with `scene_crs()` rather than assuming;
if a scene comes back in anything else, reproject before use.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from .features import SR_BANDS, scale_sr, scale_st, qa_valid_mask

BAND_ALIASES = {
    "SR_B1": "coastal", "SR_B2": "blue", "SR_B3": "green", "SR_B4": "red",
    "SR_B5": "nir", "SR_B6": "swir1", "SR_B7": "swir2",
}

SCENE_ID_RE = re.compile(
    r"(?P<sat>L[COTEM]\d{2})_(?P<proc>L2SP|L2SR)_(?P<path>\d{3})(?P<row>\d{3})_"
    r"(?P<acq>\d{8})_(?P<proc_date>\d{8})_(?P<coll>\d{2})_(?P<tier>T\d)"
)


@dataclass
class Scene:
    """A loaded Landsat scene: scaled bands, validity mask, and geo metadata."""
    scene_id: str
    bands: dict[str, np.ndarray]
    valid: np.ndarray
    transform: object          # rasterio Affine
    crs: object                # rasterio CRS
    shape: tuple[int, int]
    has_thermal: bool = True   # False for L2SR products (no ST_B10)


def find_scenes(root: str) -> list[str]:
    """Return scene directories under `root`, sorted."""
    out = [
        d for d in sorted(glob.glob(os.path.join(root, "L*_L2S*")))
        if os.path.isdir(d)
    ]
    if not out:
        raise FileNotFoundError(
            f"No Landsat Level-2 scene directories found under {root!r}. "
            "Expected untarred folders named like LC08_L2SP_127110_20200115_..."
        )
    return out


def _band_path(scene_dir: str, suffix: str) -> str:
    hits = glob.glob(os.path.join(scene_dir, f"*_{suffix}.TIF"))
    if not hits:
        raise FileNotFoundError(f"{suffix} not found in {scene_dir}")
    return hits[0]


def scene_crs(scene_dir: str):
    """Read the CRS of a scene without loading pixel data."""
    import rasterio
    with rasterio.open(_band_path(scene_dir, "QA_PIXEL")) as src:
        return src.crs


def read_scene(scene_dir: str, window=None) -> Scene:
    """
    Read and scale one Landsat Collection 2 Level-2 scene.

    `window` is an optional rasterio.windows.Window to read a subset, which is
    how you keep memory sane: a full OLI scene is ~7900 x 7900 per band, and
    eight bands at float32 is well over 2 GB once features are built.

    Pixels failing the QA mask are set NaN in every band, so downstream code
    only has to check for NaN.
    """
    import rasterio

    bands: dict[str, np.ndarray] = {}

    with rasterio.open(_band_path(scene_dir, "QA_PIXEL")) as src:
        qa = src.read(1, window=window)
        transform = src.window_transform(window) if window is not None else src.transform
        crs, shape = src.crs, qa.shape

    valid = qa_valid_mask(qa)

    for code in SR_BANDS:
        with rasterio.open(_band_path(scene_dir, code)) as src:
            arr = scale_sr(src.read(1, window=window))
        arr[~valid] = np.nan
        bands[BAND_ALIASES[code]] = arr

    # Thermal is optional. USGS issues L2SP (reflectance + surface temperature)
    # where it can and L2SR (reflectance only) where it cannot. Over Antarctica
    # the ASTER GED emissivity compilation has no valid data, so ST is not
    # generated and essentially every scene is L2SR. We proceed without it
    # rather than refusing the data, and record the fact so it can be reported.
    has_thermal = False
    try:
        with rasterio.open(_band_path(scene_dir, "ST_B10")) as src:
            st = scale_st(src.read(1, window=window))
        st[~valid] = np.nan
        bands["st_kelvin"] = st
        has_thermal = True
    except FileNotFoundError:
        pass

    return Scene(
        scene_id=os.path.basename(scene_dir.rstrip("/")),
        bands=bands, valid=valid, transform=transform, crs=crs, shape=shape,
        has_thermal=has_thermal,
    )


def pixel_coords(transform, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Projected (x, y) centre coordinates for every pixel, as 2-D arrays."""
    import rasterio.transform as rt
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    xs, ys = rt.xy(transform, rows.ravel(), cols.ravel(), offset="center")
    return (
        np.asarray(xs, dtype=np.float64).reshape(shape),
        np.asarray(ys, dtype=np.float64).reshape(shape),
    )


def parse_scene_id(scene_id: str) -> dict[str, str]:
    """Pull satellite, path/row, acquisition date out of a scene id."""
    m = SCENE_ID_RE.search(scene_id)
    return m.groupdict() if m else {}


# --- MTL metadata -----------------------------------------------------------
# Antarctic scenes are very often Tier 2, because Tier 1 requires geometric
# accuracy validated against ground control points and an ice sheet offers
# almost none: there is little stable rock to tie to and the surface moves.
# Tier 2 is spectrally fine — but these labels come from burning outcrop
# polygons onto the scene grid, so a scene georeferenced one or two pixels off
# mislabels every pixel near a boundary. Record the geometry rather than
# assume it, and widen `edge_buffer_m` if RMSE is large.

MTL_KEYS_FLOAT = (
    "GEOMETRIC_RMSE_MODEL", "GEOMETRIC_RMSE_MODEL_X", "GEOMETRIC_RMSE_MODEL_Y",
    "SUN_ELEVATION", "SUN_AZIMUTH", "CLOUD_COVER", "CLOUD_COVER_LAND",
)
MTL_KEYS_INT = ("GROUND_CONTROL_POINTS_MODEL", "GROUND_CONTROL_POINTS_VERSION")
MTL_KEYS_STR = ("COLLECTION_CATEGORY", "PROCESSING_LEVEL", "DATE_ACQUIRED", "SPACECRAFT_ID")

_MTL_LINE = re.compile(r'^\s*([A-Z0-9_]+)\s*=\s*"?([^"\n]*)"?\s*$')


def read_mtl(scene_dir: str) -> dict[str, object]:
    """
    Parse the scene MTL.txt for geometry and acquisition metadata.

    Returns a dict with whatever of MTL_KEYS_* is present. Missing keys are
    omitted rather than defaulted — in particular, GEOMETRIC_RMSE_MODEL is
    absent entirely for scenes solved from spacecraft ephemeris with no
    ground control, which is itself the signal that geolocation is weakest.
    Callers should treat a missing RMSE as "unknown and possibly poor", not
    as zero.
    """
    hits = glob.glob(os.path.join(scene_dir, "*_MTL.txt"))
    if not hits:
        return {}

    out: dict[str, object] = {}
    with open(hits[0], "r", errors="replace") as fh:
        for line in fh:
            m = _MTL_LINE.match(line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip()
            if key in MTL_KEYS_FLOAT:
                try:
                    out[key] = float(raw)
                except ValueError:
                    pass
            elif key in MTL_KEYS_INT:
                try:
                    out[key] = int(raw)
                except ValueError:
                    pass
            elif key in MTL_KEYS_STR:
                out[key] = raw
    return out


def geometry_warning(mtl: dict[str, object], pixel_size_m: float = 30.0) -> str | None:
    """
    Return a human-readable warning if scene geolocation is weak enough to
    compromise polygon-derived labels, else None.

    Thresholds are deliberately conservative: label error from misregistration
    is systematic, not random, so it biases the model rather than just adding
    noise.
    """
    rmse = mtl.get("GEOMETRIC_RMSE_MODEL")
    gcps = mtl.get("GROUND_CONTROL_POINTS_MODEL")
    tier = mtl.get("COLLECTION_CATEGORY", "?")

    if rmse is None:
        return (f"tier {tier}: no GEOMETRIC_RMSE_MODEL in MTL (likely no ground "
                f"control) — geolocation unverified, widen edge_buffer_m")
    if rmse > 2 * pixel_size_m:
        return (f"tier {tier}: geometric RMSE {rmse:.1f} m > 2 px — labels near "
                f"polygon edges unreliable, widen edge_buffer_m to >= {3 * rmse:.0f} m")
    if rmse > pixel_size_m:
        return (f"tier {tier}: geometric RMSE {rmse:.1f} m > 1 px "
                f"(gcps={gcps}) — acceptable, but check the edge buffer")
    return None
