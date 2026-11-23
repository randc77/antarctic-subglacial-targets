"""
Sampling and spatial cross-validation support.

Two problems this module exists to solve, both of which will otherwise produce
a model that scores beautifully and means nothing:

1. Class imbalance. Rock outcrop is roughly 0.18% of Antarctica
   (Burton-Johnson et al., 2016). A model that predicts "not rock" everywhere
   scores 99.8% accuracy. We stratify the training sample and report
   precision/recall/F1, never bare accuracy.

2. Spatial autocorrelation. Adjacent 30 m pixels are not independent draws.
   Random train/test splits leak neighbours across the split and inflate every
   metric. We assign pixels to square spatial blocks and split whole blocks,
   so the test set is geographically separate from the training set.

Pure NumPy: unit-testable without raster I/O.
"""

from __future__ import annotations

import numpy as np

LABEL_NONROCK, LABEL_ROCK, LABEL_EXCLUDE = 0, 1, 255


def spatial_blocks(
    x: np.ndarray, y: np.ndarray, block_km: float = 10.0
) -> np.ndarray:
    """
    Assign projected coordinates to square spatial blocks.

    Parameters
    ----------
    x, y : arrays of EPSG:3031 coordinates in metres.
    block_km : block edge length. 10 km is a reasonable default here: large
        enough to decorrelate 30 m pixels, small enough that the Prince
        Charles Mountains still yield many blocks.

    Returns
    -------
    Integer block id per point, built from the (col, row) block indices.
    """
    size = block_km * 1000.0
    bx = np.floor(x / size).astype(np.int64)
    by = np.floor(y / size).astype(np.int64)
    # Cantor-style pairing, offset so negatives behave.
    bx_ = bx - bx.min()
    by_ = by - by.min()
    return bx_ * (by_.max() + 1) + by_


def stratified_sample(
    labels: np.ndarray,
    rng: np.random.Generator,
    max_rock: int | None = None,
    nonrock_per_rock: float = 5.0,
) -> np.ndarray:
    """
    Draw a class-balanced-ish sample of row indices.

    Keeps up to `max_rock` rock pixels (all of them if None) and
    `nonrock_per_rock` times that many non-rock pixels. A 1:5 ratio keeps the
    problem honest — still imbalanced, as reality is — while making training
    tractable. Pixels labelled EXCLUDE (polygon edge buffer) are never drawn.

    Returns row indices into `labels`.
    """
    rock_idx = np.flatnonzero(labels == LABEL_ROCK)
    non_idx = np.flatnonzero(labels == LABEL_NONROCK)

    if max_rock is not None and rock_idx.size > max_rock:
        rock_idx = rng.choice(rock_idx, size=max_rock, replace=False)

    n_non = int(min(non_idx.size, round(nonrock_per_rock * rock_idx.size)))
    if n_non > 0:
        non_idx = rng.choice(non_idx, size=n_non, replace=False)
    else:
        non_idx = non_idx[:0]

    out = np.concatenate([rock_idx, non_idx])
    rng.shuffle(out)
    return out


def blocked_split(
    blocks: np.ndarray,
    rng: np.random.Generator,
    test_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split rows into train/test by whole spatial block.

    Returns boolean masks (train, test). Because blocks are assigned whole,
    no pixel in the test set is adjacent to one in the training set, which is
    the only way these metrics mean anything.
    """
    uniq = np.unique(blocks)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_fraction * uniq.size)))
    test_blocks = set(uniq[:n_test].tolist())
    is_test = np.fromiter((b in test_blocks for b in blocks), count=blocks.size, dtype=bool)
    return ~is_test, is_test


def blocked_folds(
    blocks: np.ndarray, rng: np.random.Generator, n_folds: int = 5
) -> list[np.ndarray]:
    """
    Fold assignment by whole spatial block, for spatially-blocked CV.

    Returns a list of boolean masks, one per fold, marking that fold's
    held-out rows.
    """
    uniq = np.unique(blocks)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, n_folds)
    folds = []
    for c in chunks:
        s = set(c.tolist())
        folds.append(np.fromiter((b in s for b in blocks), count=blocks.size, dtype=bool))
    return folds


def class_report_counts(labels: np.ndarray) -> dict[str, int]:
    """Quick sanity counts to print during extraction."""
    return {
        "rock": int((labels == LABEL_ROCK).sum()),
        "nonrock": int((labels == LABEL_NONROCK).sum()),
        "excluded": int((labels == LABEL_EXCLUDE).sum()),
    }
