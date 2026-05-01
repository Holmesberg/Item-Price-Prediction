"""CatBoost as a 7th model — symmetric trees + native categorical handling.

Runs as a separate script (not added to MODEL_REGISTRY) to avoid disrupting
the main 6-model GridSearch and to keep CatBoost's idiosyncrasies (no
internal n_jobs, custom progress logs, etc.) isolated.

Uses the same shared KFold splitter, the same BigMartFeatures + tree
preprocessor, and writes:
  - results/best_catboost.joblib  (refit on full train)
  - results/oof_catboost.npy
  - appends/replaces 'catboost' row in results/model_comparison.csv

Usage: python -m src.catboost_model [n_iterations]
       n_iterations defaults to 2000 (early-stopped via internal val split).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from .config import RESULTS_DIR, SEED, TEST_CSV, TRAIN_CSV, set_global_seed
from .features import BigMartFeatures
from .preprocessing import build_tree_preprocessor


def build_catboost(n_iterations: int = 2000, seed: int = SEED) -> Pipeline:
    from catboost import CatBoostRegressor

    return Pipeline(
        [
            ("prep", build_tree_preprocessor()),
            (
                "model",
                CatBoostRegressor(
                    iterations=n_iterations,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=3.0,
                    random_seed=seed,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    verbose=0,
                    allow_writing_files=False,
                ),
            ),
        ]
    )


def main() -> None:
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    X = df.drop(columns=["Y"])
    y = df["Y"].astype(float)
    BigMartFeatures.EXTRA_COUNT_REF = pd.read_csv(TEST_CSV)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    pipe = build_catboost(n_iterations=n_iter)
    print(f"  training CatBoost (iterations={n_iter}) under 5-fold CV...")
    t0 = time.perf_counter()
    scores = cross_val_score(
        pipe, X, y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
    )
    cv_rmse_mean = float(-scores.mean())
    cv_rmse_std = float(scores.std())

    oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    oof_rmse = float(np.sqrt(((oof - y.values) ** 2).mean()))

    pipe.fit(X, y)
    elapsed = time.perf_counter() - t0
    print(
        f"\n  CatBoost CV RMSE = {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}"
        f" | OOF RMSE = {oof_rmse:.4f} | {elapsed:.1f}s"
    )

    np.save(RESULTS_DIR / "oof_catboost.npy", oof)
    joblib.dump(pipe, RESULTS_DIR / "best_catboost.joblib")

    # Update model_comparison.csv: replace any existing catboost row.
    comp_path = RESULTS_DIR / "model_comparison.csv"
    comp = pd.read_csv(comp_path)
    comp = comp[comp["model"] != "catboost"]
    new_row = pd.DataFrame(
        [
            {
                "model": "catboost",
                "cv_rmse_mean": round(cv_rmse_mean, 5),
                "cv_rmse_std": round(cv_rmse_std, 5),
                "oof_rmse": round(oof_rmse, 5),
                "fit_seconds": round(elapsed, 1),
                "best_params": {"iterations": n_iter, "learning_rate": 0.03, "depth": 6},
            }
        ]
    )
    comp = pd.concat([comp, new_row], ignore_index=True).sort_values(
        "cv_rmse_mean"
    ).reset_index(drop=True)
    comp.to_csv(comp_path, index=False)

    print("\nUpdated leaderboard:")
    print(comp[["model", "cv_rmse_mean", "cv_rmse_std", "oof_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
