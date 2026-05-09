"""Train the MAE-tuned quantile model and write a calibrated submission.

Usage:
  python -m src.quantile_mae_candidate --scale 1.015 --shift 0.018
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from .config import RESULTS_DIR, ROOT, SEED, SUBMISSIONS_DIR, TEST_CSV, TRAIN_CSV
from .features import BigMartFeatures
from .models import build_quantile_mae


def _slug_float(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p")


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _calibrate(preds: np.ndarray, scale: float, shift: float) -> np.ndarray:
    center = float(preds.mean())
    return center + scale * (preds - center) + shift


def _write_submission(candidate: str, preds: np.ndarray, note: str) -> Path:
    out_dir = SUBMISSIONS_DIR / candidate
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "submission.csv"
    pd.DataFrame({"row_id": np.arange(len(preds), dtype=int), "Y": preds}).to_csv(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=0.0)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X = train_df.drop(columns=["Y"])
    y = train_df["Y"].astype(float).values

    BigMartFeatures.EXTRA_COUNT_REF = test_df
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe = build_quantile_mae(seed=SEED)

    print("== OOF check: quantile_mae ==")
    oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    cal_oof = _calibrate(oof, args.scale, args.shift)
    print(f"  base OOF MAE: {_mae(y, oof):.5f}")
    print(f"  calibrated OOF MAE: {_mae(y, cal_oof):.5f}")

    print("== Fit full train and predict test ==")
    pipe.fit(X, y)
    base_test = pipe.predict(test_df)
    preds = _calibrate(base_test, args.scale, args.shift)

    np.save(RESULTS_DIR / "oof_quantile_mae.npy", oof)
    joblib.dump(pipe, RESULTS_DIR / "best_quantile_mae.joblib")

    candidate = (
        f"cand_quantile_mae_scale{_slug_float(args.scale)}"
        f"_shift{_slug_float(args.shift)}"
    )
    note = (
        "MAE-tuned median regression candidate.\n"
        "Model: QuantileRegressor(quantile=0.5, alpha=0.001) with "
        "TargetEncoder smooth=200.\n"
        f"Affine calibration around test mean: scale={args.scale}, "
        f"shift={args.shift}."
    )
    out_csv = _write_submission(candidate, preds, note)

    summary = {
        "candidate": candidate,
        "model": "quantile_mae",
        "scale": args.scale,
        "shift": args.shift,
        "base_oof_mae": _mae(y, oof),
        "base_oof_rmse": _rmse(y, oof),
        "calibrated_oof_mae": _mae(y, cal_oof),
        "calibrated_oof_rmse": _rmse(y, cal_oof),
        "submission_csv": str(out_csv),
        "base_prediction_mean": float(base_test.mean()),
        "base_prediction_std": float(base_test.std()),
        "prediction_mean": float(preds.mean()),
        "prediction_std": float(preds.std()),
        "prediction_min": float(preds.min()),
        "prediction_max": float(preds.max()),
    }
    (RESULTS_DIR / f"{candidate}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
