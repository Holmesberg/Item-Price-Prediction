"""Train models with shared 5-fold CV + per-model GridSearch.

Outputs:
  results/model_comparison.csv  — best params, CV RMSE mean/std, fit time
  results/oof_<model>.npy       — out-of-fold predictions (in log-Y space)
  results/best_<model>.joblib   — best-of-grid pipeline, refit on full train

Usage:
  python -m src.train                  # train every model in MODEL_REGISTRY
  python -m src.train ridge_poly nn    # train only the named models, append
                                       # rows to existing model_comparison.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from .config import RESULTS_DIR, SEED, TEST_CSV, TRAIN_CSV, set_global_seed
from .features import BigMartFeatures
from .models import MODEL_REGISTRY, param_grid


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(TRAIN_CSV)
    y = df["Y"].astype(float)
    X = df.drop(columns=["Y"])
    return X, y


def main() -> None:
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_train()

    # Pool test.csv X-columns into the count-encoder statistics. Y is never
    # touched from test (only X-features), so this is a leakage-safe
    # unsupervised augmentation that gives us sharper rare-item signal.
    test_X = pd.read_csv(TEST_CSV)
    BigMartFeatures.EXTRA_COUNT_REF = test_X
    print(f"  pooling test.csv ({len(test_X)} rows) into count statistics")
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows: list[dict] = []

    requested = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_REGISTRY.keys())
    targets = {name: MODEL_REGISTRY[name] for name in requested}
    full_run = (set(requested) == set(MODEL_REGISTRY.keys()))

    for name, factory in targets.items():
        print(f"\n=== {name} ===", flush=True)
        pipe = factory(seed=SEED)
        grid = param_grid(name)

        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipe,
            grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,  # inner estimators already use n_jobs=-1
            refit=True,
            return_train_score=False,
        )
        gs.fit(X, y)
        fit_seconds = time.perf_counter() - t0

        best_idx = gs.best_index_
        cv_rmse_mean = -gs.cv_results_["mean_test_score"][best_idx]
        cv_rmse_std = gs.cv_results_["std_test_score"][best_idx]

        # OOF predictions with the best params for residual analysis + blending.
        best_pipe = factory(seed=SEED)
        best_pipe.set_params(**gs.best_params_)
        oof = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=1)
        oof_rmse = float(np.sqrt(np.mean((oof - y.values) ** 2)))

        np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
        joblib.dump(gs.best_estimator_, RESULTS_DIR / f"best_{name}.joblib")

        rows.append(
            {
                "model": name,
                "cv_rmse_mean": round(float(cv_rmse_mean), 5),
                "cv_rmse_std": round(float(cv_rmse_std), 5),
                "oof_rmse": round(oof_rmse, 5),
                "fit_seconds": round(fit_seconds, 1),
                "best_params": gs.best_params_,
            }
        )
        print(
            f"  CV RMSE = {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}"
            f"  | OOF RMSE = {oof_rmse:.4f}  | {fit_seconds:.1f}s"
            f"  | best={gs.best_params_}",
            flush=True,
        )

    new_rows = pd.DataFrame(rows)
    comp_path = RESULTS_DIR / "model_comparison.csv"
    if full_run or not comp_path.exists():
        out = new_rows
    else:
        existing = pd.read_csv(comp_path)
        existing = existing[~existing["model"].isin(new_rows["model"])]
        out = pd.concat([existing, new_rows], ignore_index=True)
    out = out.sort_values("cv_rmse_mean").reset_index(drop=True)
    out.to_csv(comp_path, index=False)
    print("\n", out.to_string(index=False))
    print(f"\nWrote {comp_path}")


if __name__ == "__main__":
    main()
