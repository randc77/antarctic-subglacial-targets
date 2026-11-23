"""
Unit tests for the pure-NumPy half of the pipeline.

These run without rasterio, geopandas or any downloaded data, so the
scientifically load-bearing logic (scaling, masking, indices, texture,
sampling, spatial splitting) can be verified before a single scene is
downloaded. Raster I/O is covered separately by integration checks that
need real data.

Run:  python -m pytest tests/ -v
      (or:  python tests/test_features.py  for a dependency-free run)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aeml import features as F          # noqa: E402
from aeml import sampling as S          # noqa: E402


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------

def test_sr_scaling_matches_usgs_constants():
    dn = np.array([[10000, 20000], [30000, F.FILL]], dtype=np.uint16)
    out = F.scale_sr(dn)
    assert np.isclose(out[0, 0], 10000 * 2.75e-05 - 0.2, atol=1e-6)
    assert np.isnan(out[1, 1]), "fill value must become NaN"


def test_st_scaling_gives_plausible_antarctic_temperatures():
    # A surface near 250 K should round-trip through the scaling.
    dn = np.array([[29550]], dtype=np.uint16)
    k = F.scale_st(dn)[0, 0]
    assert 240.0 < k < 260.0, f"expected ~250 K, got {k}"


# --------------------------------------------------------------------------
# QA masking
# --------------------------------------------------------------------------

def test_qa_mask_ignores_snow_bit():
    """
    The critical Antarctic gotcha: nearly everything is flagged snow, so a mask
    that honours the snow bit throws away the whole scene including the rock.
    """
    snow_only = np.array([[1 << F.QA_SNOW]], dtype=np.uint16)
    assert F.qa_valid_mask(snow_only)[0, 0], "snow-flagged pixels must survive"


def test_qa_mask_rejects_cloud_and_shadow_and_fill():
    for bit in (F.QA_FILL, F.QA_CLOUD, F.QA_CLOUD_SHADOW, F.QA_CIRRUS, F.QA_DILATED_CLOUD):
        qa = np.array([[1 << bit]], dtype=np.uint16)
        assert not F.qa_valid_mask(qa)[0, 0], f"bit {bit} should be masked"


# --------------------------------------------------------------------------
# indices
# --------------------------------------------------------------------------

def test_norm_diff_range_and_nan_guard():
    a = np.array([0.8, 0.0, 0.5], dtype=np.float32)
    b = np.array([0.2, 0.0, 0.5], dtype=np.float32)
    nd = F._norm_diff(a, b)
    assert np.isclose(nd[0], 0.6)
    assert np.isnan(nd[1]), "0/0 must be NaN, not inf"
    assert np.isclose(nd[2], 0.0)


def test_ndsi_separates_snow_from_rock():
    """
    Snow: bright green, dark SWIR -> high NDSI.
    Rock: moderate green, brighter SWIR -> low or negative NDSI.
    This is the signal the threshold method uses; we check our features
    reproduce it before asking a model to improve on it.
    """
    snow = {"green": np.array([[0.85]]), "swir1": np.array([[0.05]])}
    rock = {"green": np.array([[0.18]]), "swir1": np.array([[0.26]])}
    ndsi_snow = F._norm_diff(snow["green"], snow["swir1"])[0, 0]
    ndsi_rock = F._norm_diff(rock["green"], rock["swir1"])[0, 0]
    assert ndsi_snow > 0.6
    assert ndsi_rock < 0.1
    assert ndsi_snow > ndsi_rock


# --------------------------------------------------------------------------
# texture
# --------------------------------------------------------------------------

def test_local_std_zero_on_flat_field():
    flat = np.full((20, 20), 0.5, dtype=np.float32)
    assert np.allclose(F.local_std(flat, 5), 0.0, atol=1e-6)


def test_local_std_higher_on_rough_than_smooth():
    rng = np.random.default_rng(0)
    smooth = np.full((60, 60), 0.5, dtype=np.float32)
    rough = smooth + rng.normal(0, 0.1, size=smooth.shape).astype(np.float32)
    c = slice(10, 50)
    assert np.nanmean(F.local_std(rough, 5)[c, c]) > 10 * np.nanmean(F.local_std(smooth, 5)[c, c])


def test_local_std_propagates_nan_but_tolerates_neighbours():
    arr = np.full((20, 20), 0.5, dtype=np.float32)
    arr[10, 10] = np.nan
    out = F.local_std(arr, 5)
    assert np.isnan(out[10, 10]), "NaN input stays NaN"
    assert np.isfinite(out[10, 12]), "neighbour of a NaN is still computable"


# --------------------------------------------------------------------------
# feature stack
# --------------------------------------------------------------------------

def _synthetic_bands(shape=(40, 40), seed=1):
    """
    Half snow, half rock, with a NaN hole and a shadowed strip.
    Not a simulation of Antarctica, just enough structure to exercise shapes,
    NaN handling and ordering.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    is_rock = np.zeros(shape, dtype=bool)
    is_rock[:, w // 2:] = True

    def mk(snow_val, rock_val, noise=0.01):
        a = np.where(is_rock, rock_val, snow_val).astype(np.float32)
        return a + rng.normal(0, noise, shape).astype(np.float32)

    bands = {
        "coastal": mk(0.90, 0.16), "blue": mk(0.92, 0.17), "green": mk(0.88, 0.19),
        "red": mk(0.86, 0.22), "nir": mk(0.80, 0.25), "swir1": mk(0.06, 0.28),
        "swir2": mk(0.03, 0.24), "st_kelvin": mk(248.0, 266.0, noise=0.5),
    }
    bands["blue"][0:3, 0:3] = np.nan          # NaN hole
    return bands, is_rock


def test_build_features_shapes_order_and_names():
    bands, _ = _synthetic_bands()
    f = F.build_features(bands)
    assert list(f.keys()) == F.FEATURE_NAMES, "feature order must be stable"
    for k, v in f.items():
        assert v.shape == (40, 40), f"{k} wrong shape"
        assert v.dtype == np.float32, f"{k} wrong dtype"


def test_features_are_discriminative_on_synthetic_scene():
    """The stack should separate the synthetic classes; if it can't do that,
    there is no point feeding it to a classifier."""
    bands, is_rock = _synthetic_bands()
    f = F.build_features(bands)
    c = slice(5, 35)
    assert np.nanmean(f["ndsi"][c, :20]) > np.nanmean(f["ndsi"][c, 20:]) + 0.5
    assert np.nanmean(f["st_kelvin"][c, 20:]) > np.nanmean(f["st_kelvin"][c, :20]) + 10


def test_stack_to_table_drops_nan_rows_and_maps_back():
    bands, _ = _synthetic_bands()
    f = F.build_features(bands)
    valid = np.ones((40, 40), dtype=bool)
    X, idx, names = F.stack_to_table(f, valid)
    assert names == F.FEATURE_NAMES
    assert X.shape[1] == len(F.FEATURE_NAMES)
    assert X.shape[0] == idx.size
    assert np.all(np.isfinite(X)), "no NaNs may survive into the table"
    assert X.shape[0] < 1600, "NaN hole and texture edge effects should drop rows"


# --------------------------------------------------------------------------
# sampling / splitting
# --------------------------------------------------------------------------

def test_spatial_blocks_group_nearby_points():
    x = np.array([1_700_000.0, 1_700_500.0, 1_900_000.0])
    y = np.array([700_000.0, 700_500.0, 700_000.0])
    b = S.spatial_blocks(x, y, block_km=10.0)
    assert b[0] == b[1], "points 700 m apart belong in the same 10 km block"
    assert b[0] != b[2], "points 200 km apart must not"


def test_stratified_sample_respects_ratio_and_excludes():
    rng = np.random.default_rng(0)
    labels = np.concatenate([
        np.full(100, S.LABEL_ROCK),
        np.full(10_000, S.LABEL_NONROCK),
        np.full(50, S.LABEL_EXCLUDE),
    ])
    idx = S.stratified_sample(labels, rng, nonrock_per_rock=5.0)
    drawn = labels[idx]
    assert (drawn == S.LABEL_ROCK).sum() == 100
    assert (drawn == S.LABEL_NONROCK).sum() == 500
    assert (drawn == S.LABEL_EXCLUDE).sum() == 0, "buffer pixels must never be sampled"


def test_blocked_split_keeps_blocks_whole():
    rng = np.random.default_rng(3)
    blocks = np.repeat(np.arange(40), 25)
    tr, te = S.blocked_split(blocks, rng, test_fraction=0.25)
    assert not (set(blocks[tr]) & set(blocks[te])), "no block may appear on both sides"
    assert te.sum() > 0 and tr.sum() > 0


def test_blocked_folds_partition_every_row_exactly_once():
    rng = np.random.default_rng(4)
    blocks = np.repeat(np.arange(37), 11)
    folds = S.blocked_folds(blocks, rng, n_folds=5)
    counts = np.sum(np.stack(folds, axis=0), axis=0)
    assert np.all(counts == 1), "folds must partition the rows"



# --------------------------------------------------------------------------
# L2SR / no-thermal path
#
# USGS does not generate the surface-temperature product over Antarctica -
# the ASTER GED emissivity compilation has no valid data there - so scenes
# arrive as L2SR with no ST_B10 at all. The pipeline must degrade to the
# reflectance-and-texture stack rather than refuse the data.
# --------------------------------------------------------------------------

def test_feature_names_drops_only_the_thermal_pair():
    full, base = F.feature_names(True), F.feature_names(False)
    assert set(full) - set(base) == set(F.THERMAL_FEATURES)
    assert len(base) == len(full) - 2
    assert base == full[:len(base)], "thermal features must be last, so order is stable"


def test_build_features_without_thermal():
    bands, _ = _synthetic_bands()
    del bands["st_kelvin"]
    f = F.build_features(bands)
    assert list(f.keys()) == F.feature_names(False)
    assert "st_kelvin" not in f and "tex_st_std" not in f
    for k, v in f.items():
        assert v.shape == (40, 40) and v.dtype == np.float32, k


def test_no_thermal_stack_still_separates_classes():
    """Losing thermal must not lose the rock/snow signal entirely."""
    bands, _ = _synthetic_bands()
    del bands["st_kelvin"]
    f = F.build_features(bands)
    c = slice(5, 35)
    assert np.nanmean(f["ndsi"][c, :20]) > np.nanmean(f["ndsi"][c, 20:]) + 0.5
    assert np.nanmean(f["swir1"][c, 20:]) > np.nanmean(f["swir1"][c, :20]) + 0.1


def test_swir1_texture_present_in_both_stacks():
    """Added to partly compensate for the missing thermal band."""
    bands, _ = _synthetic_bands()
    assert "tex_swir1_std" in F.build_features(bands)
    del bands["st_kelvin"]
    assert "tex_swir1_std" in F.build_features(bands)


def test_stack_to_table_column_order_follows_the_dict():
    bands, _ = _synthetic_bands()
    del bands["st_kelvin"]
    f = F.build_features(bands)
    X, idx, names = F.stack_to_table(f, np.ones((40, 40), dtype=bool))
    assert names == F.feature_names(False)
    assert X.shape[1] == len(names)
    assert np.all(np.isfinite(X))



# --------------------------------------------------------------------------
# Out-of-range surface reflectance
#
# Collection 2 SR is unreliable over snow and ice: LaSRC over-corrects bright
# targets. In a real Lambert-Amery extraction, 76-84% of visible-band pixels
# came back above 1.0 and NDSI ranged from -12,198 to +17,653. These tests
# pin the clipping and the denominator guards that stop that happening.
# --------------------------------------------------------------------------

def test_scale_sr_clips_to_physical_reflectance():
    dn = np.array([[0, 7273, 30000, 43636, 50000, F.SR_DN_SATURATED]], dtype=np.uint16)
    out = F.scale_sr(dn)
    assert np.isnan(out[0, 0]), "fill -> NaN"
    assert np.isnan(out[0, -1]), "saturated -> NaN"
    finite = out[np.isfinite(out)]
    assert finite.min() >= 0.0 and finite.max() <= 1.0, "must clamp to [0, 1]"
    assert np.isclose(out[0, 2], 30000 * F.SR_SCALE + F.SR_OFFSET, atol=1e-6), \
        "in-range values must be untouched"


def test_scale_sr_clip_can_be_disabled_for_diagnostics():
    dn = np.array([[50000]], dtype=np.uint16)
    assert F.scale_sr(dn, clip=False)[0, 0] > 1.0


def test_norm_diff_is_bounded_once_inputs_are_clipped():
    """The property that failed in production: unclipped negatives blow it up."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 1, 10_000).astype(np.float32)
    b = rng.uniform(0, 1, 10_000).astype(np.float32)
    nd = F._norm_diff(a, b)
    f = np.isfinite(nd)
    assert nd[f].min() >= -1.0 and nd[f].max() <= 1.0


def test_norm_diff_rejects_near_zero_denominator():
    a = np.array([0.0004], dtype=np.float32)
    b = np.array([0.0002], dtype=np.float32)
    assert np.isnan(F._norm_diff(a, b)[0]), "both bands dark -> noise, not signal"


def test_ratio_is_capped():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([0.002], dtype=np.float32)
    assert F._ratio(a, b)[0] == 10.0, "ratio must cap, not run to 500"
    assert np.isnan(F._ratio(a, np.array([1e-5], dtype=np.float32))[0])


def test_full_stack_bounded_on_adversarial_input():
    """Feed the stack the pathological case and check nothing escapes."""
    rng = np.random.default_rng(1)
    shape = (30, 30)
    bands = {b: F.scale_sr(rng.integers(0, 65536, shape).astype(np.uint16))
             for b in ("coastal", "blue", "green", "red", "nir", "swir1", "swir2")}
    f = F.build_features(bands)
    for k in ("ndsi", "ndwi", "ndvi", "nbr"):
        v = f[k][np.isfinite(f[k])]
        assert v.min() >= -1.0 and v.max() <= 1.0, f"{k} escaped [-1, 1]"
    for k in ("swir1_nir", "swir2_swir1", "red_blue"):
        v = f[k][np.isfinite(f[k])]
        assert v.min() >= 0.0 and v.max() <= 10.0, f"{k} escaped [0, 10]"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
