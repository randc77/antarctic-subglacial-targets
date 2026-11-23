"""
Label construction from the SCAR ADD Antarctic rock-outcrop dataset.

Requires rasterio and geopandas.

The honest caveat, stated here because it belongs in the code and not only in
the paper: training on Burton-Johnson et al. (2016) polygons teaches a model
*their* definition of rock outcrop, including wherever that definition is
wrong. The model cannot be better than its labels on average — it can only be
better in specific, identifiable situations.

Two mitigations are built in:

1. An edge buffer. Pixels within `edge_buffer_m` of a polygon boundary are
   labelled EXCLUDE and never used for training or scoring. Boundary pixels
   are spectrally mixed and their label is close to a coin flip, so including
   them teaches noise and then punishes the model for having learned it.

2. Blocked spatial CV (see sampling.py), so reported skill is skill on
   unseen terrain rather than on neighbours of the training pixels.

Where the model and the benchmark disagree is the interesting output of
stage 1. Keep those pixels; do not quietly treat them as model error.
"""

from __future__ import annotations

import numpy as np

from .sampling import LABEL_EXCLUDE, LABEL_NONROCK, LABEL_ROCK


def load_outcrop(path: str, crs, bbox: tuple[float, float, float, float] | None = None):
    """
    Load the outcrop polygons, reprojected to `crs` and optionally clipped.

    `path` is the ADD rock-outcrop shapefile or geopackage.
    `bbox` is (xmin, ymin, xmax, ymax) in the target CRS.
    """
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(
            f"{path} has no CRS. The ADD outcrop layer should be EPSG:3031; "
            "set it explicitly before continuing rather than guessing."
        )
    gdf = gdf.to_crs(crs)
    if bbox is not None:
        gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return gdf


def rasterize_labels(
    outcrop_gdf,
    transform,
    shape: tuple[int, int],
    pixel_size_m: float = 30.0,
    edge_buffer_m: float = 60.0,
) -> np.ndarray:
    """
    Burn outcrop polygons to a label raster with an excluded edge band.

    Returns uint8 array of LABEL_ROCK / LABEL_NONROCK / LABEL_EXCLUDE.

    The buffer is applied symmetrically: pixels just inside and just outside
    each boundary are excluded, so the retained rock pixels are confidently
    rock and the retained non-rock pixels are confidently not.
    """
    from rasterio import features as rfeatures

    if len(outcrop_gdf) == 0:
        return np.full(shape, LABEL_NONROCK, dtype=np.uint8)

    geoms = [(geom, 1) for geom in outcrop_gdf.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        return np.full(shape, LABEL_NONROCK, dtype=np.uint8)

    rock = rfeatures.rasterize(
        geoms, out_shape=shape, transform=transform, fill=0,
        dtype=np.uint8, all_touched=False,
    ).astype(bool)

    labels = np.where(rock, LABEL_ROCK, LABEL_NONROCK).astype(np.uint8)

    if edge_buffer_m > 0:
        from scipy import ndimage
        rad = max(1, int(round(edge_buffer_m / pixel_size_m)))
        struct = ndimage.generate_binary_structure(2, 2)
        inner = ndimage.binary_erosion(rock, structure=struct, iterations=rad, border_value=0)
        outer = ndimage.binary_dilation(rock, structure=struct, iterations=rad, border_value=0)
        edge = outer & ~inner
        labels[edge] = LABEL_EXCLUDE

    return labels


def benchmark_ndsi_threshold(
    ndsi: np.ndarray,
    st_kelvin: np.ndarray | None = None,
    ndsi_max: float = 0.30,
    st_min: float = 255.0,
) -> np.ndarray:
    """
    A simple threshold-style baseline, for benchmarking.

    This is *not* the Burton-Johnson et al. (2016) algorithm - theirs is more
    careful, uses additional tests, and was tuned continent-wide. This is a
    deliberately plain stand-in so stage 1 has an honest baseline to beat in
    the same evaluation harness. When writing results up, say exactly that:
    compare against this baseline and against the published polygons, and do
    not claim to have outperformed their published method unless you have
    actually run it.

    `st_kelvin` is optional. On L2SR scenes there is no surface temperature,
    so the baseline degrades to NDSI alone - which is also what the learned
    model has to work with, keeping the comparison fair. Note in the write-up
    that the published method had thermal and this baseline does not.

    Returns a boolean "predicted rock" mask.
    """
    with np.errstate(invalid="ignore"):
        pred = ndsi < ndsi_max
        finite = np.isfinite(ndsi)
        if st_kelvin is not None:
            pred = pred & (st_kelvin > st_min)
            finite = finite & np.isfinite(st_kelvin)
    return np.where(finite, pred, False)
