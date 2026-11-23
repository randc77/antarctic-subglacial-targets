#!/usr/bin/env python3
"""
Stage 1 training and evaluation, on the corrected feature set.

Reads the .npz extraction table, repairs the derived indices by recomputing
them from reflectance clipped to [0, 1], trains a classifier with spatially
blocked cross-validation, and compares it against a threshold baseline.

Everything here is deliberately conservative about what the numbers mean:
  * splits are by whole spatial block, never by random row;
  * blocks are stratified by whether they contain rock, because most do not;
  * precision is reported at the TRUE prevalence, not the sampled one.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

NPZ = sys.argv[1] if len(sys.argv) > 1 else "data/interim/stage1_lambert_amery.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/stage1_results.json"
SEED = 20260101

# True prevalence of rock in the scenes, before stratified sampling.
# From the extraction log: 366,385 rock of 53,814,871 labelled pixels.
TRUE_PREVALENCE = 366_385 / 53_814_871

BANDS = ["coastal", "blue", "green", "red", "nir", "swir1", "swir2"]
# tex_ndsi_std was computed in image space from unclipped NDSI, so it inherits
# the blow-up and cannot be repaired from the table. Dropped here; a re-run of
# extraction with the patched code restores it.
DROP = ["tex_ndsi_std"]


def norm_diff(a, b, eps=1e-3):
    den = a + b
    out = np.full(a.shape, np.nan, np.float32)
    ok = np.abs(den) > eps
    out[ok] = (a[ok] - b[ok]) / den[ok]
    return out


def ratio(a, b, eps=1e-3, hi=10.0):
    out = np.full(a.shape, np.nan, np.float32)
    ok = np.abs(b) > eps
    out[ok] = np.clip(a[ok] / b[ok], 0.0, hi)
    return out


def rebuild(X, names):
    """Recompute point features from clipped bands; keep valid texture."""
    i = {n: k for k, n in enumerate(names)}
    B = {b: np.clip(X[:, i[b]], 0.0, 1.0) for b in BANDS}

    out = dict(B)
    out["ndsi"] = norm_diff(B["green"], B["swir1"])
    out["ndwi"] = norm_diff(B["green"], B["nir"])
    out["ndvi"] = norm_diff(B["nir"], B["red"])
    out["nbr"] = norm_diff(B["nir"], B["swir2"])
    out["swir1_nir"] = ratio(B["swir1"], B["nir"])
    out["swir2_swir1"] = ratio(B["swir2"], B["swir1"])
    out["red_blue"] = ratio(B["red"], B["blue"])
    out["brightness"] = np.mean(
        np.stack([B["blue"], B["green"], B["red"], B["nir"]]), axis=0)
    for t in ("tex_bright_std", "tex_swir1_std"):
        if t in i:
            out[t] = X[:, i[t]]

    cols = [c for c in out if c not in DROP]
    return np.column_stack([out[c] for c in cols]).astype(np.float32), cols


def blocked_split(blocks, y, rng, test_frac=0.30):
    """
    Split by whole block, stratified on whether a block contains rock.

    Most blocks contain none. A naive block split can therefore hand back a
    test set with almost no positives, and the metrics become meaningless.
    """
    uniq = np.unique(blocks)
    has_rock = np.array([y[blocks == b].sum() > 0 for b in uniq])
    test = set()
    for subset in (uniq[has_rock], uniq[~has_rock]):
        s = subset.copy()
        rng.shuffle(s)
        test.update(s[: max(1, int(round(test_frac * len(s))))].tolist())
    is_test = np.isin(blocks, list(test))
    return ~is_test, is_test


def corrected_precision(prec_sampled, sampled_prev, true_prev):
    """
    Re-express precision measured on a stratified sample at true prevalence.

    Sampling kept roughly one non-rock pixel per five; precision measured on
    that table is flattering by about the ratio of the odds. Recall and the
    ROC are unaffected by prevalence; precision is not.
    """
    if prec_sampled <= 0 or prec_sampled >= 1:
        return prec_sampled
    lr = (prec_sampled / (1 - prec_sampled)) * ((1 - sampled_prev) / sampled_prev)
    odds = lr * (true_prev / (1 - true_prev))
    return odds / (1 + odds)


def report(name, y_true, proba, sampled_prev, thr=0.5):
    pred = proba >= thr
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    prec_true = corrected_precision(prec, sampled_prev, TRUE_PREVALENCE)
    f1_true = (2 * prec_true * rec / (prec_true + rec)) if prec_true + rec else 0.0
    print(f"  {name:22s} recall {rec:6.3f}   precision {prec:6.3f} (sampled) "
          f"{prec_true:6.3f} (true prev)   F1 {f1:6.3f} / {f1_true:6.3f}")
    return dict(recall=rec, precision_sampled=prec, precision_true=prec_true,
                f1_sampled=f1, f1_true=f1_true, tp=int(tp), fp=int(fp),
                fn=int(fn), tn=int(tn))


def main() -> int:
    rng = np.random.default_rng(SEED)
    d = np.load(NPZ, allow_pickle=True)
    X0, y, names = d["X"], d["y"].astype(int), list(d["feature_names"])
    blocks, scene = d["block"], d["scene_code"]
    scene_names = [str(s) for s in d["scene_names"]]

    X, cols = rebuild(X0, names)
    keep = np.all(np.isfinite(X), axis=1)
    X, y, blocks, scene = X[keep], y[keep], blocks[keep], scene[keep]

    sampled_prev = y.mean()
    print("=" * 74)
    print("STAGE 1 - rock exposure classification, Lambert-Amery")
    print("=" * 74)
    print(f"  rows                {len(y):,}  ({100*keep.mean():.1f}% of table survived repair)")
    print(f"  features            {len(cols)}  (dropped {DROP})")
    print(f"  scenes              {len(scene_names)}")
    for k, s in enumerate(scene_names):
        m = scene == k
        print(f"      {s}   {m.sum():>9,} rows, {int(y[m].sum()):>7,} rock")
    print(f"  blocks              {len(np.unique(blocks)):,} "
          f"({int(sum(y[blocks==b].sum()>0 for b in np.unique(blocks)))} contain rock)")
    print(f"  prevalence          {sampled_prev:.3%} sampled / {TRUE_PREVALENCE:.3%} true")

    tr, te = blocked_split(blocks, y, rng)
    print(f"\n  spatially blocked split: {tr.sum():,} train / {te.sum():,} test")
    print(f"      test rock: {int(y[te].sum()):,}")

    print("\n" + "-" * 74)
    print("BASELINE  (NDSI threshold, the information a threshold method uses)")
    print("-" * 74)
    ndsi = X[:, cols.index("ndsi")]
    base_proba = (ndsi < 0.30).astype(float)
    base = report("ndsi < 0.30", y[te], base_proba[te], sampled_prev)

    print("\n" + "-" * 74)
    print("MODELS  (spatially held-out blocks)")
    print("-" * 74)
    results = {"baseline": base}
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1,
            class_weight="balanced_subsample", random_state=SEED),
        "hist_gradient_boost": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=SEED),
    }
    fitted = {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        r = report(name, y[te], proba, sampled_prev)
        r["roc_auc"] = float(roc_auc_score(y[te], proba))
        r["avg_precision"] = float(average_precision_score(y[te], proba))
        print(f"  {'':22s} ROC-AUC {r['roc_auc']:.4f}   "
              f"avg precision {r['avg_precision']:.4f}")
        results[name] = r
        fitted[name] = (m, proba)

    best = max(fitted, key=lambda k: results[k]["roc_auc"])
    model, proba = fitted[best]
    print(f"\n  best by ROC-AUC: {best}")

    print("\n" + "-" * 74)
    print("OPERATING POINTS  (best model, precision at true prevalence)")
    print("-" * 74)
    p, r, t = precision_recall_curve(y[te], proba)
    for target in (0.50, 0.70, 0.80, 0.90, 0.95):
        k = int(np.argmin(np.abs(r[:-1] - target)))
        pt = corrected_precision(p[k], sampled_prev, TRUE_PREVALENCE)
        print(f"  recall {r[k]:.2f}  ->  threshold {t[k]:.3f}   "
              f"precision {p[k]:.3f} sampled / {pt:.3f} true")

    print("\n" + "-" * 74)
    print("FEATURE IMPORTANCE  (permutation, on held-out blocks)")
    print("-" * 74)
    sub = rng.choice(np.flatnonzero(te), size=min(40_000, int(te.sum())), replace=False)
    pi = permutation_importance(model, X[sub], y[sub], n_repeats=5,
                                random_state=SEED, n_jobs=-1, scoring="roc_auc")
    order = np.argsort(pi.importances_mean)[::-1]
    for k in order:
        bar = "#" * int(60 * pi.importances_mean[k] / max(pi.importances_mean.max(), 1e-9))
        print(f"  {cols[k]:16s} {pi.importances_mean[k]:7.4f} +/- "
              f"{pi.importances_std[k]:.4f}  {bar}")

    if len(scene_names) > 1:
        print("\n" + "-" * 74)
        print("LEAVE-ONE-SCENE-OUT  (does it generalise to a new footprint?)")
        print("-" * 74)
        for k, s in enumerate(scene_names):
            trs, tes = scene != k, scene == k
            if y[tes].sum() < 100 or y[trs].sum() < 100:
                print(f"  {s[:34]:34s} skipped (too few rock pixels)")
                continue
            m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1,
                                       class_weight="balanced_subsample", random_state=SEED)
            m.fit(X[trs], y[trs])
            pr = m.predict_proba(X[tes])[:, 1]
            report(f"held out {s[5:17]}", y[tes], pr, sampled_prev)

    with open(OUT, "w") as fh:
        json.dump({"features": cols, "true_prevalence": TRUE_PREVALENCE,
                   "sampled_prevalence": float(sampled_prev),
                   "n_rows": int(len(y)), "results": results}, fh, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
