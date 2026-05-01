"""Train models on the decomposed target Y - log1p(X6) (predicted log-volume),
then add log1p(X6) back at predict-time to recover Y predictions.

Hypothesis: Y = log(MRP × Volume) = log(MRP) + log(Volume). By subtracting
log(MRP) from the target we leave only the log-volume component, which has
less variance and may be easier for tree-based models to fit. The predictions
get log(MRP) added back at test time.

Runs the full 5-model GridSearch on the decomposed target. CV scores are
reported in the **original Y space** (we add log1p(X6) back inside the
scoring loop), so they're directly comparable to the standard run. Saves
candidates to results/best_<name>_decomp.joblib + oof_<name>_decomp.npy.

Usage: python -m src.decomp_experiment [model1] [model2] ...
       Defaults to ['ridge', 'lgbm'] (the two strongest base models).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

from .config import RESULTS_DIR, SEED, TRAIN_CSV, set_global_seed
from .models import MODEL_REGISTRY, param_grid


def main() -> None:
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    requested = sys.argv[1:] if len(sys.argv) > 1 else ["ridge", "lgbm"]
    df = pd.read_csv(TRAIN_CSV)
    y = df["Y"].astype(float).values
    X = df.drop(columns=["Y"])
    log1p_x6 = np.log1p(X["X6"].astype(float).values)
    y_decomp = y - log1p_x6

    print(f"  Y mean: {y.mean():.4f}, std: {y.std():.4f}")
    print(f"  log1p(X6) mean: {log1p_x6.mean():.4f}, std: {log1p_x6.std():.4f}")
    print(f"  Y_decomp mean: {y_decomp.mean():.4f}, std: {y_decomp.std():.4f}")
    print(f"  Decomposed target has {y_decomp.std() / y.std():.2%} of original Y std.")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name in requested:
        if name not in MODEL_REGISTRY:
            print(f"  unknown model {name!r} — skipping")
            continue
        print(f"\n=== {name} on Y_decomp ===")

        # Train on Y_decomp via GridSearchCV
        pipe = MODEL_REGISTRY[name](seed=SEED)
        grid = param_grid(name)
        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipe, grid,
            scoring="neg_root_mean_squared_error",
            cv=cv, n_jobs=1, refit=True,
        )
        gs.fit(X, y_decomp)
        # Note: gs.best_score_ is RMSE on Y_decomp (decomposed space).

        # Recompute CV RMSE in **original Y space** by re-running CV with
        # log1p(X6) added back to predictions.
        best_pipe = MODEL_REGISTRY[name](seed=SEED)
        best_pipe.set_params(**gs.best_params_)
        oof_decomp = cross_val_predict(best_pipe, X, y_decomp, cv=cv, n_jobs=1)
        oof_y = oof_decomp + log1p_x6
        oof_rmse_y = float(np.sqrt(((oof_y - y) ** 2).mean()))
        elapsed = time.perf_counter() - t0

        cv_decomp_mean = -gs.cv_results_["mean_test_score"][gs.best_index_]
        cv_decomp_std = gs.cv_results_["std_test_score"][gs.best_index_]
        print(
            f"  CV(Y_decomp) = {cv_decomp_mean:.4f} ± {cv_decomp_std:.4f}"
            f"  | OOF RMSE in Y space = {oof_rmse_y:.4f} | {elapsed:.1f}s"
            f"  | best={gs.best_params_}"
        )

        # Persist OOF (in Y-space) and best estimator (refit on Y_decomp).
        np.save(RESULTS_DIR / f"oof_{name}_decomp.npy", oof_y)
        joblib.dump(gs.best_estimator_, RESULTS_DIR / f"best_{name}_decomp.joblib")

        rows.append(
            {
                "model": f"{name}_decomp",
                "cv_rmse_mean": round(oof_rmse_y, 5),  # in Y space
                "cv_rmse_std": round(float(cv_decomp_std), 5),
                "oof_rmse": round(oof_rmse_y, 5),
                "fit_seconds": round(elapsed, 1),
                "best_params": gs.best_params_,
            }
        )

    # Update model_comparison.csv in-place: replace rows for any *_decomp models.
    comp_path = RESULTS_DIR / "model_comparison.csv"
    new_rows = pd.DataFrame(rows)
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        comp = comp[~comp["model"].isin(new_rows["model"])]
        comp = pd.concat([comp, new_rows], ignore_index=True).sort_values(
            "cv_rmse_mean"
        ).reset_index(drop=True)
    else:
        comp = new_rows.sort_values("cv_rmse_mean").reset_index(drop=True)
    comp.to_csv(comp_path, index=False)

    print("\nUpdated leaderboard (RMSE in original Y space):")
    print(comp[["model", "cv_rmse_mean", "cv_rmse_std", "oof_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
