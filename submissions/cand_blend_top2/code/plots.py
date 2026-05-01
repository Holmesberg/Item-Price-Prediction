"""Generate diagnostic figures for the report.

  - LightGBM feature importance (gain)
  - Predicted vs actual (using OOF predictions of the top model)
  - Residual histogram

Usage: python -m src.plots
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, RESULTS_DIR, TRAIN_CSV, set_global_seed, SEED
from .features import ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC


def _feature_names_after_prep(pipe) -> list[str]:
    """Best-effort feature names out of the tree preprocessor."""
    encode = pipe.named_steps["prep"].named_steps["encode"]
    try:
        return list(encode.get_feature_names_out())
    except Exception:
        return ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL


def plot_feature_importance() -> None:
    pipe = joblib.load(RESULTS_DIR / "best_lgbm.joblib")
    booster = pipe.named_steps["model"].booster_
    names = _feature_names_after_prep(pipe)
    gains = booster.feature_importance(importance_type="gain")
    if len(names) != len(gains):
        names = [f"f{i}" for i in range(len(gains))]
    s = pd.Series(gains, index=names).sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(7, 5))
    s.plot(kind="barh", ax=ax, color="#3f72af")
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM feature importance (top 15)")
    fig.tight_layout()
    out = FIGURES_DIR / "feature_importance_lgbm.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_residuals() -> None:
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    top_model = comp.sort_values("cv_rmse_mean").iloc[0]["model"]
    oof = np.load(RESULTS_DIR / f"oof_{top_model}.npy")
    y = pd.read_csv(TRAIN_CSV)["Y"].values
    resid = y - oof

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(oof, y, s=6, alpha=0.4, color="#112d4e")
    lo, hi = float(min(oof.min(), y.min())), float(max(oof.max(), y.max()))
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[0].set_xlabel("Predicted (log Y)")
    axes[0].set_ylabel("Actual (log Y)")
    axes[0].set_title(f"Predicted vs actual — {top_model} (OOF)")

    axes[1].hist(resid, bins=40, color="#3f72af", edgecolor="white")
    axes[1].axvline(0, color="r", linewidth=1)
    axes[1].set_xlabel("Residual (actual − predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residuals — {top_model} (OOF)")

    fig.tight_layout()
    out = FIGURES_DIR / "residuals_top_model.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    set_global_seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance()
    plot_residuals()


if __name__ == "__main__":
    main()
