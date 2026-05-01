"""Optuna TPE sweep for XGBoost — mirrors src.optuna_lgbm.

Usage: python -m src.optuna_xgb [n_trials]
       n_trials defaults to 50.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

from .config import RESULTS_DIR, SEED, TEST_CSV, TRAIN_CSV, set_global_seed
from .features import BigMartFeatures
from .models import build_xgb


def _objective(trial: optuna.Trial, X, y, cv) -> float:
    params = {
        "model__n_estimators": trial.suggest_int("n_estimators", 200, 4000, log=True),
        "model__learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.2, log=True),
        "model__max_depth": trial.suggest_int("max_depth", 3, 10),
        "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "model__gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    pipe = build_xgb(seed=SEED)
    pipe.set_params(**params)
    scores = cross_val_score(
        pipe, X, y,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=1,
    )
    return -float(scores.mean())


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    X = df.drop(columns=["Y"])
    y = df["Y"].astype(float)
    BigMartFeatures.EXTRA_COUNT_REF = pd.read_csv(TEST_CSV)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    t0 = time.perf_counter()
    study.optimize(
        lambda t: _objective(t, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"\nOptuna XGB done in {elapsed:.1f}s | "
        f"best CV RMSE = {study.best_value:.4f} | "
        f"best params = {study.best_params}"
    )

    best_pipe_params = {f"model__{k}": v for k, v in study.best_params.items()}
    best_pipe = build_xgb(seed=SEED)
    best_pipe.set_params(**best_pipe_params)
    oof = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=1)
    oof_rmse = float(np.sqrt(((oof - y.values) ** 2).mean()))
    cv_rmse_std = float(
        np.std([
            -s for s in cross_val_score(
                best_pipe, X, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1
            )
        ])
    )
    best_pipe.fit(X, y)

    np.save(RESULTS_DIR / "oof_xgb.npy", oof)
    joblib.dump(best_pipe, RESULTS_DIR / "best_xgb.joblib")
    joblib.dump(study, RESULTS_DIR / "optuna_xgb_study.joblib")

    comp_path = RESULTS_DIR / "model_comparison.csv"
    comp = pd.read_csv(comp_path)
    comp = comp[comp["model"] != "xgb"]
    new_row = pd.DataFrame(
        [
            {
                "model": "xgb",
                "cv_rmse_mean": round(study.best_value, 5),
                "cv_rmse_std": round(cv_rmse_std, 5),
                "oof_rmse": round(oof_rmse, 5),
                "fit_seconds": round(elapsed, 1),
                "best_params": study.best_params,
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
