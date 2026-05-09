"""Generate calibrated submissions from saved joblib models.

This is for late-stage MAE leaderboard probes where an older saved model has
useful OOF behavior but is not part of the slim final training registry.

Usage:
  python -m src.joblib_calibration --model huber --scale 1.025 --shift 0.04
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    RESULTS_DIR,
    ROOT,
    SUBMISSIONS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    Y_LOG_MAX,
    Y_LOG_MIN,
)
from .features import BigMartFeatures


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
    parser.add_argument("--model", required=True, help="Saved model name, e.g. huber")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=0.0)
    args = parser.parse_args()

    model_path = RESULTS_DIR / f"best_{args.model}.joblib"
    oof_path = RESULTS_DIR / f"oof_{args.model}.npy"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    y = train_df["Y"].astype(float).values
    BigMartFeatures.EXTRA_COUNT_REF = test_df

    model = joblib.load(model_path)
    base_test = model.predict(test_df)
    preds = _calibrate(base_test, args.scale, args.shift)

    metrics: dict[str, float] = {}
    if oof_path.exists():
        base_oof = np.load(oof_path)
        if len(base_oof) == len(y):
            cal_oof = _calibrate(base_oof, args.scale, args.shift)
            metrics = {
                "base_oof_mae": _mae(y, base_oof),
                "base_oof_rmse": _rmse(y, base_oof),
                "calibrated_oof_mae": _mae(y, cal_oof),
                "calibrated_oof_rmse": _rmse(y, cal_oof),
            }

    lo, hi = float(preds.min()), float(preds.max())
    if lo < Y_LOG_MIN or hi > Y_LOG_MAX:
        print(
            f"  WARN preds out of expected log-Y range: [{lo:.2f}, {hi:.2f}] "
            f"(expected roughly [{Y_LOG_MIN}, {Y_LOG_MAX}])"
        )
    else:
        print(f"  preds range OK: [{lo:.2f}, {hi:.2f}]")

    candidate = (
        f"cand_{args.model}_scale{_slug_float(args.scale)}"
        f"_shift{_slug_float(args.shift)}"
    )
    note = (
        f"Legacy joblib candidate: {args.model}.\n"
        f"Affine calibration around test mean: scale={args.scale}, "
        f"shift={args.shift}.\n"
        "Generated for MAE leaderboard probing after confirming the competition "
        "metric is Mean Absolute Error."
    )
    out_csv = _write_submission(candidate, preds, note)

    summary = {
        "candidate": candidate,
        "model": args.model,
        "scale": args.scale,
        "shift": args.shift,
        **metrics,
        "submission_csv": str(out_csv),
        "base_prediction_mean": float(base_test.mean()),
        "base_prediction_std": float(base_test.std()),
        "prediction_mean": float(preds.mean()),
        "prediction_std": float(preds.std()),
        "prediction_min": lo,
        "prediction_max": hi,
    }
    out_json = RESULTS_DIR / f"{candidate}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
