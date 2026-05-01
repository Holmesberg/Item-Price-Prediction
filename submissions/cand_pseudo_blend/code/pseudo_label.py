"""Pseudo-labeling: use the current best stacked ensemble to predict test
labels, take the high-confidence subset (low ensemble standard deviation),
add them as pseudo-labels to the training data, and retrain Ridge on the
augmented set.

The retrained Ridge produces TEST predictions only (no aligned OOF since
the training set is augmented). It's used as an additional column at
blend time, not as a base model in the OOF stack.

Usage: python -m src.pseudo_label [confidence_quantile]
       confidence_quantile defaults to 0.5 (use the most-confident half).

Output: results/pseudo_ridge_test_pred.npy (predictions on test, log-Y space)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import RESULTS_DIR, SEED, TEST_CSV, TRAIN_CSV, set_global_seed
from .features import BigMartFeatures
from .models import build_ridge


def _load_base_test_preds(test_df: pd.DataFrame, exclude=("ridge_poly",)):
    """Load every best_*.joblib (skip ridge_poly which is highly correlated
    with ridge), call predict on test, return dict {name: preds}."""
    out = {}
    for path in sorted(RESULTS_DIR.glob("best_*.joblib")):
        name = path.stem.replace("best_", "")
        if name in exclude:
            continue
        if "_decomp" in name or "_pseudo" in name:
            continue
        pipe = joblib.load(path)
        out[name] = pipe.predict(test_df)
    return out


def main() -> None:
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    BigMartFeatures.EXTRA_COUNT_REF = test
    X_tr, y_tr = train.drop(columns=["Y"]), train["Y"].astype(float)

    print("Loading base test predictions...")
    base_test = _load_base_test_preds(test)
    if not base_test:
        raise RuntimeError("no best_*.joblib files found — train models first")
    names = list(base_test.keys())
    print(f"  using base models: {names}")
    test_stack = np.column_stack([base_test[n] for n in names])

    # Mean prediction = pseudo-label; std across models = inverse confidence.
    mean_pred = test_stack.mean(axis=1)
    std_pred = test_stack.std(axis=1)
    threshold = float(np.quantile(std_pred, q))
    confident = std_pred <= threshold
    print(
        f"  using top {q:.0%} most-confident test rows ({confident.sum()}/{len(test)})"
        f" — std threshold {threshold:.4f}"
    )

    test_pseudo = test.loc[confident].copy()
    y_pseudo = pd.Series(mean_pred[confident], index=test_pseudo.index, name="Y")

    X_aug = pd.concat([X_tr, test_pseudo], ignore_index=True)
    y_aug = pd.concat([y_tr, y_pseudo], ignore_index=True)
    print(f"  augmented training: {X_tr.shape[0]} train + {confident.sum()} pseudo = {X_aug.shape[0]}")

    print("\nRetraining Ridge on augmented data...")
    t0 = time.perf_counter()
    pipe = build_ridge(seed=SEED)
    pipe.set_params(model__alpha=10.0)
    pipe.fit(X_aug, y_aug)
    elapsed = time.perf_counter() - t0
    pseudo_test_pred = pipe.predict(test)
    print(f"  retrained in {elapsed:.1f}s | preds range [{pseudo_test_pred.min():.3f}, {pseudo_test_pred.max():.3f}]")

    out_pred = RESULTS_DIR / "pseudo_ridge_test_pred.npy"
    np.save(out_pred, pseudo_test_pred)
    joblib.dump(pipe, RESULTS_DIR / "best_ridge_pseudo.joblib")
    print(f"  wrote {out_pred} + best_ridge_pseudo.joblib")

    # Quick comparison: how different are pseudo-Ridge preds from base Ridge?
    base_ridge = base_test.get("ridge")
    if base_ridge is not None:
        diff = pseudo_test_pred - base_ridge
        print(
            f"\n  pseudo-Ridge vs base-Ridge on test:"
            f" mean diff {diff.mean():+.4f}, std diff {diff.std():.4f},"
            f" |max| diff {np.abs(diff).max():.4f}"
        )


if __name__ == "__main__":
    main()
