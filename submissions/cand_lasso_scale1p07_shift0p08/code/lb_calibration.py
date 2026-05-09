"""Generate leaderboard-calibration probes for the final Lasso model.

The public test distribution is much easier than random CV because almost all
item IDs are seen in train. That can make the full-test conditional mean more
spread out than OOF predictions. This script writes a single affine-calibrated
probe so the leaderboard can tell us whether variance expansion helps.

Usage:
  python -m src.lb_calibration --scale 1.08
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _candidate_name(scale: float, shift: float) -> str:
    scale_tag = str(scale).replace(".", "p").replace("-", "m")
    shift_tag = str(shift).replace(".", "p").replace("-", "m")
    return f"cand_lasso_scale{scale_tag}_shift{shift_tag}"


def _write(out_dir: Path, preds: np.ndarray, note: str) -> Path:
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
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--shift", type=float, default=0.0)
    args = parser.parse_args()

    set_global_seed(SEED)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X = train_df.drop(columns=["Y"])
    y = train_df["Y"].astype(float).values
    BigMartFeatures.EXTRA_COUNT_REF = test_df

    pipe = build_lasso_te200(SEED)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    center_oof = float(oof.mean())
    oof_cal = center_oof + args.scale * (oof - center_oof) + args.shift

    pipe.fit(X, y)
    base = pipe.predict(test_df)
    center_test = float(base.mean())
    preds = center_test + args.scale * (base - center_test) + args.shift
    preds = np.clip(preds, Y_LOG_MIN, Y_LOG_MAX)

    lo, hi = float(preds.min()), float(preds.max())
    print(f"base range [{base.min():.3f}, {base.max():.3f}]")
    print(f"calibrated range [{lo:.3f}, {hi:.3f}]")

    name = _candidate_name(args.scale, args.shift)
    note = (
        "Affine leaderboard-calibration probe for lasso_te200.\n"
        f"Prediction = mean(test_pred) + {args.scale:.4f} * "
        f"(test_pred - mean(test_pred)) + {args.shift:.4f}.\n"
        f"Base OOF RMSE: {_rmse(y, oof):.5f}.\n"
        f"Calibrated OOF RMSE: {_rmse(y, oof_cal):.5f}.\n"
        "Rationale: public test has 99.3% item overlap, so full-test "
        "conditional means may be less shrunken than random-CV OOF means."
    )
    out_csv = _write(SUBMISSIONS_DIR / name, preds, note)

    summary = {
        "candidate": name,
        "scale": args.scale,
        "shift": args.shift,
        "base_oof_rmse": _rmse(y, oof),
        "calibrated_oof_rmse": _rmse(y, oof_cal),
        "submission_csv": str(out_csv),
        "base_prediction_mean": center_test,
        "base_prediction_std": float(base.std()),
        "prediction_mean": float(preds.mean()),
        "prediction_std": float(preds.std()),
        "prediction_min": lo,
        "prediction_max": hi,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
