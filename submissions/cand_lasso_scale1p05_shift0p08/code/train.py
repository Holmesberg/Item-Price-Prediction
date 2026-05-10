"""Train the final <=5-model lineup with shared 5-fold CV.

Outputs:
  results/model_comparison.csv  - OOF MAE/RMSE summary, fit time
  results/oof_<model>.npy       - out-of-fold predictions in Y space
  results/best_<model>.joblib   - final tuned pipeline, refit on full train

Usage:
  python -m src.train               # train the final lineup
  python -m src.train lasso_te200   # train only one named model
"""

from __future__ import annotations

import sys
import time

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
    test_X = pd.read_csv(TEST_CSV)
    BigMartFeatures.EXTRA_COUNT_REF = test_X
    print(f"  pooling test.csv ({len(test_X)} rows) into count statistics")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(MODEL_REGISTRY.keys())
    targets = {name: MODEL_REGISTRY[name] for name in requested}
    full_run = set(requested) == set(MODEL_REGISTRY.keys())

    rows: list[dict] = []
    for name, factory in targets.items():
        print(f"\n=== {name} ===", flush=True)
        pipe = factory(seed=SEED)

        t0 = time.perf_counter()
        gs = GridSearchCV(
            pipe,
            param_grid(name),
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X, y)
        fit_seconds = time.perf_counter() - t0

        best_pipe = factory(seed=SEED)
        best_pipe.set_params(**gs.best_params_)
        oof = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=1)
        oof_mae = float(np.mean(np.abs(oof - y.values)))
        oof_rmse = float(np.sqrt(np.mean((oof - y.values) ** 2)))

        np.save(RESULTS_DIR / f"oof_{name}.npy", oof)
        joblib.dump(gs.best_estimator_, RESULTS_DIR / f"best_{name}.joblib")

        rows.append(
            {
                "model": name,
                "oof_mae": round(oof_mae, 5),
                "oof_rmse": round(oof_rmse, 5),
                "fit_seconds": round(fit_seconds, 1),
                "best_params": gs.best_params_,
            }
        )
        print(
            f"  OOF MAE = {oof_mae:.4f}  | OOF RMSE = {oof_rmse:.4f}"
            f"  | {fit_seconds:.1f}s | best={gs.best_params_}",
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

    out = out.sort_values("oof_mae").reset_index(drop=True)
    out.to_csv(comp_path, index=False)
    print("\n", out.to_string(index=False))
    print(f"\nWrote {comp_path}")


if __name__ == "__main__":
    main()
