"""Generate the minimal final leaderboard candidate.

The repo already contains a broad model zoo from exploration. This script is
for the final, low-submission phase: it trains one legal single-model variant
that improved OOF in the May 9 sweep, writes its OOF/full-fit artifacts, and
creates a reproducible Kaggle submission directory.

Usage:
  python -m src.final_lb_candidates
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

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
from .models import build_lasso_te200

CANDIDATE_NAME = "cand_lasso_te200"
MODEL_NAME = "lasso_te200"
KNOWN_PUBLIC_LB = 0.378
SUBMITTED_AT = "2026-05-09 06:55:22 UTC"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _sanity(preds: np.ndarray, n_expected: int) -> None:
    assert preds.shape == (n_expected,), f"got {preds.shape}, expected ({n_expected},)"
    assert not np.any(np.isnan(preds)), "NaN in predictions"
    lo, hi = float(preds.min()), float(preds.max())
    if lo < Y_LOG_MIN or hi > Y_LOG_MAX:
        print(
            f"  WARN preds out of expected log-Y range: [{lo:.2f}, {hi:.2f}] "
            f"(expected roughly [{Y_LOG_MIN}, {Y_LOG_MAX}])"
        )
    else:
        print(f"  preds range OK: [{lo:.2f}, {hi:.2f}]")


def _write_submission(preds: np.ndarray, n_test: int, note: str) -> Path:
    out_dir = SUBMISSIONS_DIR / CANDIDATE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "submission.csv"
    pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": preds}).to_csv(
        out_csv, index=False
    )
    (out_dir / "note.txt").write_text(note + "\n", encoding="utf-8")

    code_snapshot = out_dir / "code"
    if code_snapshot.exists():
        shutil.rmtree(code_snapshot)
    shutil.copytree(
        ROOT / "src",
        code_snapshot,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "requirements.txt", out_dir / "requirements.txt")
    return out_csv


def main() -> None:
    set_global_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X = train_df.drop(columns=["Y"])
    y = train_df["Y"].astype(float).values

    # The count features pool test X values only; no target leakage.
    BigMartFeatures.EXTRA_COUNT_REF = test_df
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    pipe = build_lasso_te200(seed=SEED)
    print("== OOF check: lasso_te200 ==")
    oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    oof_rmse = _rmse(y, oof)
    print(f"  OOF RMSE: {oof_rmse:.5f}")

    print("== Fit full train and predict test ==")
    pipe.fit(X, y)
    preds = pipe.predict(test_df)
    _sanity(preds, len(test_df))

    np.save(RESULTS_DIR / f"oof_{MODEL_NAME}.npy", oof)
    joblib.dump(pipe, RESULTS_DIR / f"best_{MODEL_NAME}.joblib")

    note = (
        "Single-model final candidate under the 5-model cap.\n"
        "Model: ElasticNet used as Lasso (alpha=0.003, l1_ratio=1.0) with "
        "TargetEncoder smooth=200.\n"
        f"OOF RMSE: {oof_rmse:.5f}.\n"
        f"Public LB: {KNOWN_PUBLIC_LB:.3f} (submitted {SUBMITTED_AT}).\n"
        "Chosen because it beat the previous ENet single-model OOF "
        "(0.52069) without adding another model family or a risky stack."
    )
    out_csv = _write_submission(preds, len(test_df), note)

    summary = {
        "candidate": CANDIDATE_NAME,
        "model": MODEL_NAME,
        "models_used": 1,
        "oof_rmse": oof_rmse,
        "submission_csv": str(out_csv),
        "prediction_mean": float(preds.mean()),
        "prediction_std": float(preds.std()),
        "prediction_min": float(preds.min()),
        "prediction_max": float(preds.max()),
        "public_lb_score": KNOWN_PUBLIC_LB,
        "submitted_at": SUBMITTED_AT,
    }
    (RESULTS_DIR / "final_lb_candidates.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
