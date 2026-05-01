"""Stacking with multiple meta-learner variants.

When base-model OOF correlations are high (here: 0.97-0.999), plain
linear-stack weights can fail to spread weight across diverse models. This
script tries:
  - Constrained NNLS (non-negative weights, sum-to-1 enforced via projection)
  - LightGBM meta-learner (non-linear blend)
  - Mean of all
  - Geometric mean
  - Simple average of top-k by OOF RMSE

Each variant emits a candidate submission under submissions/.

Usage: python -m src.stacking_v2
"""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

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
from .features import BigMartFeatures


def _load_oofs() -> tuple[list[str], np.ndarray]:
    paths = sorted(p for p in RESULTS_DIR.glob("oof_*.npy") if "_decomp" not in p.stem)
    names = [p.stem.replace("oof_", "") for p in paths]
    cols = [np.load(p) for p in paths]
    return names, np.column_stack(cols)


def _load_test_stack(test_df, names) -> np.ndarray:
    cols = []
    for n in names:
        pipe = joblib.load(RESULTS_DIR / f"best_{n}.joblib")
        cols.append(pipe.predict(test_df))
    return np.column_stack(cols)


def _nnls_constrained(oof_stack, y) -> np.ndarray:
    """Find weights w >= 0, sum(w) = 1, minimizing ||Xw - y||^2."""
    n_models = oof_stack.shape[1]
    init = np.ones(n_models) / n_models

    def loss(w):
        return float(np.mean((oof_stack @ w - y) ** 2))

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * n_models
    res = minimize(loss, init, method="SLSQP", constraints=cons, bounds=bounds)
    return res.x


def _write(name, preds, n_test, note):
    out = SUBMISSIONS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": preds}).to_csv(
        out / "submission.csv", index=False
    )
    (out / "note.txt").write_text(note + "\n", encoding="utf-8")
    code = out / "code"
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(
        ROOT / "src", code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")


def main() -> None:
    set_global_seed(SEED)
    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)
    BigMartFeatures.EXTRA_COUNT_REF = test_df

    y = pd.read_csv(TRAIN_CSV)["Y"].astype(float).values
    names, oof_stack = _load_oofs()
    print(f"Base models: {names}")
    test_stack = _load_test_stack(test_df, names)

    # ----- NNLS-constrained stacking -----
    w_nnls = _nnls_constrained(oof_stack, y)
    print("\nNNLS-constrained weights (>=0, sum=1):")
    for n, w in zip(names, w_nnls):
        print(f"  {n:<10} {w:.4f}")
    oof_nnls = oof_stack @ w_nnls
    rmse_nnls = float(np.sqrt(((oof_nnls - y) ** 2).mean()))
    test_nnls = test_stack @ w_nnls
    print(f"  OOF RMSE: {rmse_nnls:.5f}")
    _write("cand_stack_nnls", test_nnls, n_test,
           note=f"NNLS-constrained meta. OOF RMSE: {rmse_nnls:.5f}\n"
                + "Weights:\n" + "\n".join(f"  {n:<10} {w:.4f}" for n, w in zip(names, w_nnls)))

    # ----- LightGBM meta-learner -----
    try:
        from lightgbm import LGBMRegressor
        meta_lgbm = LGBMRegressor(
            n_estimators=200, learning_rate=0.03, num_leaves=15,
            min_child_samples=20, random_state=SEED, verbosity=-1, n_jobs=1,
        )
        meta_lgbm.fit(oof_stack, y)
        oof_lgbm_meta = meta_lgbm.predict(oof_stack)
        rmse_lgbm_meta = float(np.sqrt(((oof_lgbm_meta - y) ** 2).mean()))
        test_lgbm_meta = meta_lgbm.predict(test_stack)
        print(f"\nLGBM meta-learner OOF RMSE: {rmse_lgbm_meta:.5f}")
        print(
            "  Note: LGBM meta-learner OOF score is overfit "
            "(meta sees the same OOFs it was trained on); "
            "test predictions are still valid."
        )
        _write("cand_stack_lgbm", test_lgbm_meta, n_test,
               note=f"LightGBM meta-learner. (Self-fit OOF RMSE {rmse_lgbm_meta:.5f} "
                    "is over-optimistic.)")
    except Exception as e:
        print(f"  LGBM meta failed: {e}")

    # ----- Geometric mean (in raw-sales space, then back to log) -----
    # mean(log(Y)) over models = log(geomean(exp(log(Y)))) — so this is
    # equivalent to arithmetic mean in log-Y space. No new candidate.

    # ----- Powered mean: mean of top-k -----
    # Top-3 by OOF RMSE
    rmses = [(n, float(np.sqrt(((oof_stack[:, i] - y) ** 2).mean())))
             for i, n in enumerate(names)]
    top3 = [n for n, _ in sorted(rmses, key=lambda x: x[1])[:3]]
    idx_top3 = [names.index(n) for n in top3]
    oof_top3 = oof_stack[:, idx_top3].mean(axis=1)
    test_top3 = test_stack[:, idx_top3].mean(axis=1)
    rmse_top3 = float(np.sqrt(((oof_top3 - y) ** 2).mean()))
    print(f"\nMean-of-top-3 ({top3}) OOF RMSE: {rmse_top3:.5f}")
    _write("cand_mean_top3", test_top3, n_test,
           note=f"Equal-weight mean of top-3 by OOF RMSE: {top3}\nOOF RMSE: {rmse_top3:.5f}")

    print("\nDone — candidates in submissions/")


if __name__ == "__main__":
    main()
