"""
Spectral and textural feature construction for Landsat 8/9 Collection 2 Level-2.

Pure NumPy/SciPy: no raster I/O here, so this module is unit-testable without
GDAL, rasterio or any downloaded scene. Raster reading lives in `landsat.py`.

Antarctic-specific notes
------------------------
* The QA_PIXEL *snow* bit is useless here. Almost every pixel in the study
  region is flagged snow, including rock that happens to sit next to a drift.
  We mask only fill, cloud, cirrus and cloud shadow.
* Surface temperature (ST_B10) is the single most discriminating band in this
  setting. Sunlit rock runs well above the surrounding snow, which is pinned
  near the melting point. Burton-Johnson et al. (2016) leaned on TIRS for
  exactly this reason.
* Dark, debris-covered and shadowed ice is where threshold methods struggle,
  which is where a learned model has room to beat them. Features are chosen to
  give the classifier something to work with in those cases: thermal, texture,
  and SWIR behaviour rather than brightness alone.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# --- Landsat Collection 2 Level-2 scaling -----------------------------------
# https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
SR_SCALE, SR_OFFSET = 2.75e-05, -0.2       # surface reflectance -> 0-1
SR_DN_SATURATED = 65535                    # top of the uint16 range
ST_SCALE, ST_OFFSET = 0.00341802, 149.0    # surface temperature -> kelvin
FILL = 0

# Collection 2 QA_PIXEL bit positions
QA_FILL, QA_DILATED_CLOUD, QA_CIRRUS, QA_CLOUD, QA_CLOUD_SHADOW = 0, 1, 2, 3, 4
QA_SNOW, QA_CLEAR, QA_WATER = 5, 6, 7

SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

# Thermal features are listed LAST and kept separate, because USGS does not
# generate the Level-2 surface-temperature product everywhere. Over Antarctica
# the ASTER GED emissivity compilation has no valid data, so scenes arrive as
# L2SR (reflectance only) rather than L2SP, and ST_B10 simply does not exist.
# The pipeline therefore has to work both with and without thermal.
FEATURE_NAMES_BASE = [
    # scaled surface reflectance
    "blue", "green", "red", "nir", "swir1", "swir2", "coastal",
    # normalised indices
    "ndsi", "ndwi", "ndvi", "nbr",
    # ratios and composites
    "swir1_nir", "swir2_swir1", "red_blue", "brightness",
    # texture
    "tex_bright_std", "tex_ndsi_std", "tex_swir1_std",
]

THERMAL_FEATURES = ["st_kelvin", "tex_st_std"]

FEATURE_NAMES = FEATURE_NAMES_BASE + THERMAL_FEATURES


def feature_names(thermal: bool = True) -> list[str]:
    """Feature names in stack order, with or without the thermal pair."""
    return list(FEATURE_NAMES) if thermal else list(FEATURE_NAMES_BASE)



def scale_sr(dn: np.ndarray, clip: bool = True) -> np.ndarray:
    """
    Scale Collection 2 surface-reflectance DN to reflectance.

    Fill and saturated DN become NaN. With `clip` (the default) the result is
    then clamped to physical reflectance, [0, 1].

    Why the clip is not optional in practice
    ----------------------------------------
    USGS cautions that Collection 2 surface reflectance is unreliable over
    snow and ice: LaSRC's aerosol retrieval assumes dark, dense-vegetation
    targets and over-corrects on bright ones. Over the Lambert-Amery region
    the consequence is severe - in a real extraction, 76-84% of visible-band
    pixels came back above 1.0 and 1-3% came back negative.

    Unclipped, that destroys every derived index. A normalised difference
    (a-b)/(a+b) is bounded to [-1, 1] only while a and b are non-negative;
    let one go negative and the denominator can approach zero while the
    numerator does not. Measured on real data, NDSI ranged from -12,198 to
    +17,653 before this clip was added.

    Clipping costs within-snow variation in the visible bands, which is
    saturated and carries little information anyway. It preserves the
    rock/snow discrimination, which lives in SWIR and in the indices.
    """
    out = dn.astype(np.float32) * SR_SCALE + SR_OFFSET
    out[dn == FILL] = np.nan
    out[dn >= SR_DN_SATURATED] = np.nan
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def scale_st(dn: np.ndarray) -> np.ndarray:
    """Scale Collection 2 surface-temperature DN to kelvin, fill -> NaN."""
    out = dn.astype(np.float32) * ST_SCALE + ST_OFFSET
    out[dn == FILL] = np.nan
    return out


def qa_valid_mask(qa: np.ndarray) -> np.ndarray:
    """
    True where a pixel is usable.

    Deliberately ignores the snow bit (see module docstring) and the water bit:
    shelf and sea ice are excluded later by the study-region mask, not here,
    because QA water is unreliable over fast ice.
    """
    bad = (
        ((qa >> QA_FILL) & 1).astype(bool)
        | ((qa >> QA_DILATED_CLOUD) & 1).astype(bool)
        | ((qa >> QA_CIRRUS) & 1).astype(bool)
        | ((qa >> QA_CLOUD) & 1).astype(bool)
        | ((qa >> QA_CLOUD_SHADOW) & 1).astype(bool)
    )
    return ~bad


def _norm_diff(a: np.ndarray, b: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Normalised difference (a - b) / (a + b), guarded against divide-by-zero.

    eps is 1e-3, not machine epsilon: with reflectance clipped to [0, 1], a
    denominator below 1e-3 means both bands are essentially dark, and the
    ratio there is noise rather than signal.
    """
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(denom) & (np.abs(denom) > eps)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def _ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-3,
           hi: float = 10.0) -> np.ndarray:
    """
    Band ratio a / b, guarded and capped.

    Capped at `hi` because a reflectance ratio above ~10 means the
    denominator band is near zero, where the quotient is dominated by noise.
    Leaving it uncapped puts long tails into the feature distribution that
    tree splits will happily chase.
    """
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > eps)
    out[ok] = np.clip(a[ok] / b[ok], 0.0, hi)
    return out


def local_std(arr: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Local standard deviation in a square window, NaN-aware.

    Texture matters here because outcrop is rough at 30 m while snow and blue
    ice are smooth. A shadowed rock face and a shadowed snow slope can share a
    reflectance value but not a texture.
    """
    finite = np.isfinite(arr)
    filled = np.where(finite, arr, 0.0).astype(np.float64)
    w = finite.astype(np.float64)

    n = ndimage.uniform_filter(w, size=size, mode="nearest")
    s = ndimage.uniform_filter(filled, size=size, mode="nearest")
    s2 = ndimage.uniform_filter(filled * filled, size=size, mode="nearest")

    out = np.full(arr.shape, np.nan, dtype=np.float32)
    ok = n > 0.5 / (size * size)          # require at least one valid neighbour
    mean = np.zeros_like(s)
    mean[ok] = s[ok] / n[ok]
    var = np.zeros_like(s)
    var[ok] = np.maximum(s2[ok] / n[ok] - mean[ok] ** 2, 0.0)
    out[ok] = np.sqrt(var[ok]).astype(np.float32)
    out[~finite] = np.nan
    return out


def build_features(bands: dict[str, np.ndarray], texture_size: int = 5) -> dict[str, np.ndarray]:
    """
    Build the stage-1 feature stack from scaled band arrays.

    Thermal is optional. If `bands` contains 'st_kelvin' the two thermal
    features are appended; if it does not (an L2SR scene), they are omitted
    and the stack is the reflectance-and-texture set alone. The returned dict
    is ordered, and its keys are the authoritative column order downstream -
    do not assume FEATURE_NAMES.

    Parameters
    ----------
    bands : dict
        Scaled arrays keyed 'coastal','blue','green','red','nir','swir1',
        'swir2', and optionally 'st_kelvin'. Same shape, NaN for invalid.
    texture_size : int
        Window edge length in pixels for local-std texture (5 -> 150 m).
    """
    b = bands
    thermal = "st_kelvin" in b and b["st_kelvin"] is not None
    f: dict[str, np.ndarray] = {}

    for k in ("blue", "green", "red", "nir", "swir1", "swir2", "coastal"):
        f[k] = b[k].astype(np.float32)

    # Normalised indices. NDSI is the workhorse of the threshold approach;
    # included so the model has access to the same information, not because
    # we rely on it.
    f["ndsi"] = _norm_diff(b["green"], b["swir1"])
    f["ndwi"] = _norm_diff(b["green"], b["nir"])
    f["ndvi"] = _norm_diff(b["nir"], b["red"])
    f["nbr"] = _norm_diff(b["nir"], b["swir2"])

    f["swir1_nir"] = _ratio(b["swir1"], b["nir"])
    f["swir2_swir1"] = _ratio(b["swir2"], b["swir1"])
    f["red_blue"] = _ratio(b["red"], b["blue"])

    brightness = np.nanmean(
        np.stack([b["blue"], b["green"], b["red"], b["nir"]], axis=0), axis=0
    ).astype(np.float32)
    f["brightness"] = brightness

    f["tex_bright_std"] = local_std(brightness, texture_size)
    f["tex_ndsi_std"] = local_std(f["ndsi"], texture_size)
    # SWIR1 texture partly compensates for a missing thermal band: SWIR is
    # where rock and snow separate most strongly in reflectance, so structure
    # in it carries information that brightness texture alone does not.
    f["tex_swir1_std"] = local_std(b["swir1"].astype(np.float32), texture_size)

    if thermal:
        f["st_kelvin"] = b["st_kelvin"].astype(np.float32)
        f["tex_st_std"] = local_std(b["st_kelvin"], texture_size)

    return {k: f[k] for k in feature_names(thermal)}


def stack_to_table(
    features: dict[str, np.ndarray], valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Flatten a feature dict to an (n_valid, n_features) matrix.

    Column order comes from the dict itself, so a thermal-free stack works
    without special-casing. Returns (X, flat_index, names): flat_index records
    which flattened pixel each surviving row came from, and rows containing
    any NaN are dropped.
    """
    names = list(features.keys())
    cols = [features[k].ravel() for k in names]
    X = np.column_stack(cols).astype(np.float32)
    keep = valid.ravel() & np.all(np.isfinite(X), axis=1)
    return X[keep], np.flatnonzero(keep), names
