"""Generate the top-2 reproducible submissions.

  sub_01_<top>     — single best model by CV RMSE.
  sub_02_blend     — inverse-CV-RMSE-weighted blend of the top model and its
                      least-correlated strong peer (chosen from OOF correlations
                      across the next 3 strongest models). Data-driven, not hardcoded.

Each submission directory holds the CSV plus a frozen snapshot of `src/`
(without __pycache__) so the submission stays reproducible after later edits.

Usage: python -m src.submit
"""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    RESULTS_DIR,
    ROOT,
    SEED,
    SUBMISSIONS_DIR,
    TEST_CSV,
    Y_LOG_MAX,
    Y_LOG_MIN,
    set_global_seed,
)


def _load_best(name: str):
    path = RESULTS_DIR / f"best_{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m src.train` first.")
    return joblib.load(path)


def _load_oof(name: str) -> np.ndarray:
    path = RESULTS_DIR / f"oof_{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `python -m src.train` first.")
    return np.load(path)


def _sanity(preds: np.ndarray, n_expected: int) -> None:
    assert preds.shape == (n_expected,), f"got {preds.shape}, expected ({n_expected},)"
    assert not np.any(np.isnan(preds)), "NaN in predictions"
    lo, hi = float(preds.min()), float(preds.max())
    if lo < Y_LOG_MIN or hi > Y_LOG_MAX:
        print(
            f"  WARN preds out of expected log-Y range: [{lo:.2f}, {hi:.2f}] "
            f"(expected ~[{Y_LOG_MIN}, {Y_LOG_MAX}])"
        )
    else:
        print(f"  preds range OK: [{lo:.2f}, {hi:.2f}]")


def _write(submission_dir: Path, preds: np.ndarray, n_test: int) -> None:
    submission_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": preds})
    out = submission_dir / "submission.csv"
    df.to_csv(out, index=False)

    code_snapshot = submission_dir / "code"
    if code_snapshot.exists():
        shutil.rmtree(code_snapshot)
    shutil.copytree(
        ROOT / "src",
        code_snapshot,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "requirements.txt", submission_dir / "requirements.txt")

    print(f"  wrote {out} ({len(df)} rows) + code snapshot")


def _pick_blend_partner(top: str, comp: pd.DataFrame) -> tuple[str, float]:
    """Return (partner_name, oof_corr_with_top). Partner = least-correlated of
    the next 3 strongest models (by CV RMSE). If fewer than 3 peers exist,
    pick from whatever's available."""
    oof_top = _load_oof(top)
    candidates = [m for m in comp["model"].tolist() if m != top][:3]
    if not candidates:
        raise RuntimeError("no peer models available for blending")
    corrs = {
        m: float(np.corrcoef(oof_top, _load_oof(m))[0, 1])
        for m in candidates
    }
    print("  OOF correlations vs top:")
    for m, c in corrs.items():
        print(f"    {m:<8} corr={c:+.3f}")
    partner = min(corrs, key=corrs.get)
    return partner, corrs[partner]


def main() -> None:
    set_global_seed(SEED)
    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)

    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv").sort_values(
        "cv_rmse_mean"
    ).reset_index(drop=True)
    print("Model leaderboard (CV RMSE):")
    print(comp[["model", "cv_rmse_mean", "cv_rmse_std"]].to_string(index=False))

    top = comp.iloc[0]["model"]
    rmse_top = float(comp.iloc[0]["cv_rmse_mean"])
    pipe_top = _load_best(top)

    print(f"\n== Submission 1: best single model ({top}, CV RMSE {rmse_top:.4f}) ==")
    p_top = pipe_top.predict(test_df)
    _sanity(p_top, n_test)
    _write(SUBMISSIONS_DIR / f"sub_01_{top}", p_top, n_test)

    print("\n== Submission 2: OOF-correlation-driven blend ==")
    partner, corr = _pick_blend_partner(top, comp)
    rmse_partner = float(comp.set_index("model").loc[partner, "cv_rmse_mean"])
    pipe_partner = _load_best(partner)
    p_partner = pipe_partner.predict(test_df)

    w_top = (1.0 / rmse_top) / (1.0 / rmse_top + 1.0 / rmse_partner)
    w_partner = 1.0 - w_top
    p_blend = w_top * p_top + w_partner * p_partner

    print(
        f"  blend = {w_top:.3f} * {top} + {w_partner:.3f} * {partner}"
        f"  (corr={corr:+.3f})"
    )
    _sanity(p_blend, n_test)
    _write(SUBMISSIONS_DIR / f"sub_02_blend_{top}_{partner}", p_blend, n_test)


if __name__ == "__main__":
    main()
