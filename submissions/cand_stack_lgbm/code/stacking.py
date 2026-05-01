"""Stacked ensemble: Ridge meta-learner on OOF preds of all base models.

Pipeline:
  1. Load OOF preds for all base models in results/oof_*.npy.
  2. Stack into (n_train, n_models) matrix.
  3. Fit Ridge(alpha=...) on (OOF, y) — this is what produces the meta-weights.
  4. For test predictions: predict test with each base pipe, stack into
     (n_test, n_models), pass through the trained Ridge meta-learner.
  5. Sanity-check + write submission.

The Ridge meta-learner is a constrained generalization of inverse-RMSE
weighting: it picks weights that minimize RMSE on the OOF stack. It's
robust to highly correlated base preds (which is our regime: OOF corrs
~0.97–0.99) because Ridge handles multicollinearity via L2 reg.

Usage: python -m src.stacking
Output: submissions/sub_03_stack/submission.csv + frozen code snapshot.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

from .config import (
    RESULTS_DIR,
    ROOT,
    SEED,
    SUBMISSIONS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    Y_LOG_MAX,
    Y_LOG_MIN,
    set_global_seed,
)


def _load_oof_matrix() -> tuple[list[str], np.ndarray]:
    paths = sorted(RESULTS_DIR.glob("oof_*.npy"))
    if not paths:
        raise FileNotFoundError("no oof_*.npy files in results/")
    names = [p.stem.replace("oof_", "") for p in paths]
    cols = [np.load(p) for p in paths]
    return names, np.column_stack(cols)


def _load_best(name: str):
    path = RESULTS_DIR / f"best_{name}.joblib"
    return joblib.load(path)


def main() -> None:
    set_global_seed(SEED)
    df_y = pd.read_csv(TRAIN_CSV)["Y"].astype(float).values

    names, oof_stack = _load_oof_matrix()
    print(f"Stacking base models: {names}")
    print(f"OOF stack shape: {oof_stack.shape}")

    # OOF correlation matrix — sanity print to confirm diversity.
    oof_corr = np.corrcoef(oof_stack.T)
    print("\nOOF correlation matrix:")
    print(pd.DataFrame(oof_corr, index=names, columns=names).round(3).to_string())

    # Fit RidgeCV on (OOF, y) to find the meta-learner alpha.
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(oof_stack, df_y)
    coefs = dict(zip(names, meta.coef_))
    print(f"\nMeta-learner alpha={meta.alpha_:.3f}, intercept={meta.intercept_:.4f}")
    print("Meta weights (model -> coef):")
    for n in names:
        print(f"  {n:<10} {coefs[n]:+.4f}")

    oof_blend = meta.predict(oof_stack)
    oof_rmse = float(np.sqrt(((oof_blend - df_y) ** 2).mean()))
    print(f"\nStacked OOF RMSE: {oof_rmse:.5f}")

    # Predict test by stacking each base model's test predictions.
    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)
    test_stack_cols = []
    for name in names:
        pipe = _load_best(name)
        p = pipe.predict(test_df)
        assert p.shape == (n_test,)
        test_stack_cols.append(p)
    test_stack = np.column_stack(test_stack_cols)

    test_pred = meta.predict(test_stack)
    assert test_pred.shape == (n_test,)
    assert not np.any(np.isnan(test_pred))
    lo, hi = float(test_pred.min()), float(test_pred.max())
    print(f"Test pred range: [{lo:.2f}, {hi:.2f}] (expected ~[{Y_LOG_MIN}, {Y_LOG_MAX}])")

    out_dir = SUBMISSIONS_DIR / "sub_03_stack"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": test_pred}).to_csv(
        out_dir / "submission.csv", index=False
    )
    code_snapshot = out_dir / "code"
    if code_snapshot.exists():
        shutil.rmtree(code_snapshot)
    shutil.copytree(
        ROOT / "src",
        code_snapshot,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "requirements.txt", out_dir / "requirements.txt")
    print(f"\nwrote {out_dir / 'submission.csv'} + code snapshot")


if __name__ == "__main__":
    main()
