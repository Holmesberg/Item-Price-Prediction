"""Generate every reasonable submission candidate from the trained base models.

Outputs (under submissions/):
  cand_single_<top>           best single model by CV
  cand_blend_invrmse          inverse-CV-RMSE blend of top-K models
  cand_blend_top2             inverse-CV-RMSE blend of top-2 by CV
  cand_blend_uncorr           inverse-CV-RMSE blend of top model + least-correlated peer
  cand_stack_ridge            Ridge meta-learner stacked on all OOFs
  cand_mean                   simple arithmetic mean of all base preds
  cand_mean_top4              arithmetic mean of top-4 by CV
  cand_geomean_top4           geometric mean (in raw-sales space) of top-4

Each candidate gets a directory with submission.csv + a small note.txt
describing how it was produced + its OOF-RMSE estimate. We pick best 2 to
upload to Kaggle tomorrow when the daily limit resets.

Usage: python -m src.all_candidates
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

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
    """Load every oof_*.npy except _decomp variants (those are in Y space too,
    but we keep them as a separate group)."""
    paths = sorted(p for p in RESULTS_DIR.glob("oof_*.npy") if "_decomp" not in p.stem)
    names = [p.stem.replace("oof_", "") for p in paths]
    cols = [np.load(p) for p in paths]
    return names, np.column_stack(cols)


def _load_pipe(name: str):
    return joblib.load(RESULTS_DIR / f"best_{name}.joblib")


def _y_train() -> np.ndarray:
    return pd.read_csv(TRAIN_CSV)["Y"].astype(float).values


def _write_candidate(name: str, preds: np.ndarray, n_test: int, note: str) -> None:
    out_dir = SUBMISSIONS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": preds}).to_csv(
        out_dir / "submission.csv", index=False
    )
    (out_dir / "note.txt").write_text(note + "\n", encoding="utf-8")
    code_snapshot = out_dir / "code"
    if code_snapshot.exists():
        shutil.rmtree(code_snapshot)
    shutil.copytree(
        ROOT / "src", code_snapshot,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "requirements.txt", out_dir / "requirements.txt")


def _sanity(preds: np.ndarray, label: str) -> None:
    assert not np.any(np.isnan(preds)), f"{label}: NaN in preds"
    lo, hi = float(preds.min()), float(preds.max())
    if lo < Y_LOG_MIN or hi > Y_LOG_MAX:
        print(f"  WARN {label}: preds range [{lo:.2f},{hi:.2f}] outside expected log-Y")


def main() -> None:
    set_global_seed(SEED)
    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)
    y_tr = _y_train()
    BigMartFeatures.EXTRA_COUNT_REF = test_df

    names, oof_stack = _load_oofs()
    print(f"Base models: {names}")
    print(f"OOF stack shape: {oof_stack.shape}")

    # Per-model metadata
    per = {}
    for i, n in enumerate(names):
        oof = oof_stack[:, i]
        per[n] = {
            "rmse": float(np.sqrt(((oof - y_tr) ** 2).mean())),
            "oof": oof,
            "test_pred": _load_pipe(n).predict(test_df),
        }

    test_stack = np.column_stack([per[n]["test_pred"] for n in names])

    # OOF correlations
    oof_corr = pd.DataFrame(np.corrcoef(oof_stack.T), index=names, columns=names)
    print("\nOOF correlation matrix:")
    print(oof_corr.round(3).to_string())

    sorted_by_rmse = sorted(per.items(), key=lambda kv: kv[1]["rmse"])
    top_name = sorted_by_rmse[0][0]
    print(f"\nTop single model: {top_name} (OOF RMSE {per[top_name]['rmse']:.4f})")

    # ----- cand_single_<top> -----
    p = per[top_name]["test_pred"]
    _sanity(p, f"single_{top_name}")
    _write_candidate(
        f"cand_single_{top_name}", p, n_test,
        note=f"Best single model by OOF RMSE: {top_name} ({per[top_name]['rmse']:.4f})",
    )

    # ----- cand_blend_invrmse: inverse-RMSE blend of all models -----
    weights = np.array([1.0 / per[n]["rmse"] for n in names])
    weights /= weights.sum()
    blend_test = sum(w * per[n]["test_pred"] for w, n in zip(weights, names))
    blend_oof = sum(w * per[n]["oof"] for w, n in zip(weights, names))
    blend_rmse = float(np.sqrt(((blend_oof - y_tr) ** 2).mean()))
    _sanity(blend_test, "blend_invrmse")
    _write_candidate(
        "cand_blend_invrmse", blend_test, n_test,
        note=(
            f"Inverse-OOF-RMSE blend of all {len(names)} base models.\n"
            + f"OOF RMSE: {blend_rmse:.4f}\nWeights:\n"
            + "\n".join(f"  {n:<10} {w:.4f}" for w, n in zip(weights, names))
        ),
    )

    # ----- cand_blend_top2: inverse-RMSE blend of top-2 -----
    top2 = [sorted_by_rmse[0][0], sorted_by_rmse[1][0]]
    w2 = np.array([1.0 / per[n]["rmse"] for n in top2])
    w2 /= w2.sum()
    blend_top2 = sum(w * per[n]["test_pred"] for w, n in zip(w2, top2))
    oof_top2 = sum(w * per[n]["oof"] for w, n in zip(w2, top2))
    rmse_top2 = float(np.sqrt(((oof_top2 - y_tr) ** 2).mean()))
    _sanity(blend_top2, "blend_top2")
    _write_candidate(
        "cand_blend_top2", blend_top2, n_test,
        note=f"Top-2 inverse-RMSE blend: {top2[0]} ({w2[0]:.3f}) + {top2[1]} ({w2[1]:.3f})\n"
             f"OOF RMSE: {rmse_top2:.4f}",
    )

    # ----- cand_blend_uncorr: top + least-correlated of next 3 strongest -----
    next3 = [sorted_by_rmse[i][0] for i in range(1, min(4, len(sorted_by_rmse)))]
    corrs = {n: float(np.corrcoef(per[top_name]["oof"], per[n]["oof"])[0, 1]) for n in next3}
    partner = min(corrs, key=corrs.get)
    pair = [top_name, partner]
    wpair = np.array([1.0 / per[n]["rmse"] for n in pair])
    wpair /= wpair.sum()
    blend_uncorr = sum(w * per[n]["test_pred"] for w, n in zip(wpair, pair))
    oof_uncorr = sum(w * per[n]["oof"] for w, n in zip(wpair, pair))
    rmse_uncorr = float(np.sqrt(((oof_uncorr - y_tr) ** 2).mean()))
    _sanity(blend_uncorr, "blend_uncorr")
    _write_candidate(
        "cand_blend_uncorr", blend_uncorr, n_test,
        note=f"Top + least-correlated of next 3: {pair[0]} ({wpair[0]:.3f}) + {pair[1]} ({wpair[1]:.3f})\n"
             f"OOF corr: {corrs[partner]:+.3f}\nOOF RMSE: {rmse_uncorr:.4f}",
    )

    # ----- cand_stack_ridge: Ridge meta-learner on full OOF stack -----
    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(oof_stack, y_tr)
    stack_oof = meta.predict(oof_stack)
    stack_test = meta.predict(test_stack)
    stack_rmse = float(np.sqrt(((stack_oof - y_tr) ** 2).mean()))
    weights_meta = dict(zip(names, meta.coef_))
    _sanity(stack_test, "stack_ridge")
    _write_candidate(
        "cand_stack_ridge", stack_test, n_test,
        note=(
            f"RidgeCV meta-learner on full OOF stack.\n"
            f"Meta alpha: {meta.alpha_:.4f}\nOOF RMSE: {stack_rmse:.4f}\n"
            f"Intercept: {meta.intercept_:+.4f}\nMeta weights:\n"
            + "\n".join(f"  {n:<10} {w:+.4f}" for n, w in weights_meta.items())
        ),
    )

    # ----- cand_mean: arithmetic mean of all -----
    mean_test = test_stack.mean(axis=1)
    mean_oof = oof_stack.mean(axis=1)
    rmse_mean = float(np.sqrt(((mean_oof - y_tr) ** 2).mean()))
    _sanity(mean_test, "mean")
    _write_candidate(
        "cand_mean", mean_test, n_test,
        note=f"Equal-weight mean of all {len(names)} models.\nOOF RMSE: {rmse_mean:.4f}",
    )

    # ----- cand_mean_top4 + cand_geomean_top4 -----
    top4 = [sorted_by_rmse[i][0] for i in range(min(4, len(sorted_by_rmse)))]
    sub = np.column_stack([per[n]["test_pred"] for n in top4])
    sub_oof = np.column_stack([per[n]["oof"] for n in top4])
    mean4 = sub.mean(axis=1)
    mean4_oof = sub_oof.mean(axis=1)
    rmse_mean4 = float(np.sqrt(((mean4_oof - y_tr) ** 2).mean()))
    _sanity(mean4, "mean_top4")
    _write_candidate(
        "cand_mean_top4", mean4, n_test,
        note=f"Equal-weight mean of top-4: {top4}\nOOF RMSE: {rmse_mean4:.4f}",
    )

    # geomean in raw-sales space: exp(mean(Y_log)) — equivalent to mean in Y_log
    # space exactly! So "geomean in raw" = arithmetic mean in our log-Y. Skip.

    # Print the summary table
    print("\nCandidate summary (OOF RMSE in log-Y space):")
    summary = [
        ("cand_single_" + top_name, per[top_name]["rmse"]),
        ("cand_blend_invrmse", blend_rmse),
        ("cand_blend_top2", rmse_top2),
        ("cand_blend_uncorr", rmse_uncorr),
        ("cand_stack_ridge", stack_rmse),
        ("cand_mean", rmse_mean),
        ("cand_mean_top4", rmse_mean4),
    ]
    summary.sort(key=lambda x: x[1])
    for name, rmse in summary:
        print(f"  {name:<30}  {rmse:.5f}")
    (RESULTS_DIR / "candidates_summary.json").write_text(
        json.dumps([{"name": n, "oof_rmse": r} for n, r in summary], indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
