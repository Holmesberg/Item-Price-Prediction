"""Generate the EDA figures the report needs.

Run once. Produces 7 figures in report/figures/.

Usage: python -m src.eda
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, TRAIN_CSV, TEST_CSV, set_global_seed, SEED


def _save(fig, name: str) -> None:
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    set_global_seed(SEED)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    # 1) Target distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(train["Y"], bins=40, kde=True, ax=ax, color="#3f72af")
    ax.set_title("Target Y distribution (already log-transformed)")
    ax.set_xlabel("Y (log sales)")
    _save(fig, "01_target_dist.png")

    # 2) Missingness
    miss = pd.DataFrame(
        {
            "train_pct": train.isna().mean() * 100,
            "test_pct": test.isna().mean() * 100,
        }
    )
    miss = miss[(miss > 0).any(axis=1)].sort_values("train_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    miss.plot(kind="barh", ax=ax, color=["#3f72af", "#dbe2ef"])
    ax.set_xlabel("% missing")
    ax.set_title("Missingness — train vs test")
    _save(fig, "02_missingness.png")

    # 3) X3 dirty values
    fig, ax = plt.subplots(figsize=(7, 3.5))
    train["X3"].value_counts().plot(kind="bar", ax=ax, color="#3f72af")
    ax.set_title("X3 raw value counts (5 dirty variants)")
    ax.set_xlabel("X3 value")
    ax.set_ylabel("Count")
    _save(fig, "03_x3_dirty.png")

    # 4) X4 zero anomaly
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(train["X4"], bins=60, ax=ax, color="#3f72af")
    n_zero = int((train["X4"] == 0).sum())
    ax.axvline(0, color="red", linewidth=1)
    ax.set_title(f"X4 (item visibility) — {n_zero} mislabeled zeros")
    ax.set_xlabel("X4")
    _save(fig, "04_x4_zero_anomaly.png")

    # 5) X6 vs Y
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(train["X6"], train["Y"], s=6, alpha=0.3, color="#112d4e")
    ax.set_xlabel("X6 (Item MRP)")
    ax.set_ylabel("Y (log sales)")
    ax.set_title("X6 vs Y — strongest single predictor")
    _save(fig, "05_x6_vs_y.png")

    # 6) X1 prefix vs Y
    train_prefix = train.assign(X1_prefix=train["X1"].str.slice(0, 2))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(
        data=train_prefix, x="X1_prefix", y="Y", hue="X1_prefix",
        ax=ax, palette="Blues", legend=False,
    )
    ax.set_title("Y by X1 prefix (FD=food, DR=drink, NC=non-consumable)")
    _save(fig, "06_x1_prefix_vs_y.png")

    # 7) Outlet rollups
    train_year = train.assign(Outlet_Years=2013 - train["X8"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(
        data=train_year, x="X11", y="Y", hue="X11",
        ax=axes[0], palette="Blues", legend=False,
    )
    axes[0].set_title("Y by Outlet Type (X11)")
    axes[0].tick_params(axis="x", rotation=20)
    sns.boxplot(
        data=train_year, x="Outlet_Years", y="Y", hue="Outlet_Years",
        ax=axes[1], palette="Blues", legend=False,
    )
    axes[1].set_title("Y by Outlet Years (2013 − X8)")
    _save(fig, "07_outlet_rollups.png")

    # 8) Correlation heatmap
    num = train[["X2", "X4", "X6", "X8", "Y"]].copy()
    num.loc[num["X4"] == 0, "X4"] = np.nan
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric correlation (X4 zeros → NaN)")
    _save(fig, "08_correlation.png")


if __name__ == "__main__":
    main()
