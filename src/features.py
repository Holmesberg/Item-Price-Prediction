"""Domain-aware feature engineering for the BigMart dataset.

All transforms are sklearn-compatible and `fit` only on the training fold so
nothing leaks across the CV split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import OUTLET_REF_YEAR

RAW_NUMERIC = ["X2", "X4", "X6", "X8"]
RAW_CATEGORICAL = ["X1", "X3", "X5", "X7", "X9", "X10", "X11"]

# Engineered output columns after BigMartFeatures.transform:
ENGINEERED_NUMERIC = ["X2", "X4", "X6", "Outlet_Years", "X4_ratio"]
ENGINEERED_CATEGORICAL = ["X1_prefix", "X3", "X5", "X6_bin", "X7", "X9", "X10", "X11"]


class BigMartFeatures(BaseEstimator, TransformerMixin):
    """Domain-aware preprocessing.

    Steps:
      1. X1_prefix from first 2 chars of X1 (FD/DR/NC).
      2. X3 normalized to {Low Fat, Regular}; Non-Edible override when X1_prefix==NC.
      3. X4: 0 -> NaN, then group-mean impute on X1 with global fallback.
      4. X2: group-mean impute on X1 with global fallback.
      5. X6_bin: 4 quantile bins on X6 (kept alongside raw X6).
      6. Outlet_Years = OUTLET_REF_YEAR - X8.
      7. X9: mode-impute within X11.
      8. X4_ratio = X4 / mean(X4 by X7).
      9. Drop X1 and X8 (X1 too high-cardinality, X8 replaced by Outlet_Years).
    """

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X.loc[X["X4"] == 0, "X4"] = np.nan

        self.x1_x4_mean_ = X.groupby("X1")["X4"].mean()
        self.x1_x2_mean_ = X.groupby("X1")["X2"].mean()
        self.global_x4_mean_ = float(X["X4"].mean())
        self.global_x2_mean_ = float(X["X2"].mean())

        x4_imputed = self._impute_by_group(
            X["X1"], X["X4"], self.x1_x4_mean_, self.global_x4_mean_
        )
        tmp = X.assign(X4=x4_imputed)
        self.x7_x4_mean_ = tmp.groupby("X7")["X4"].mean()
        self.global_x4_imputed_mean_ = float(x4_imputed.mean())

        self.x6_bin_edges_ = np.quantile(
            X["X6"].dropna().values, [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        self.x6_bin_edges_[0] -= 1e-6
        self.x6_bin_edges_[-1] += 1e-6

        self.x11_x9_mode_ = (
            X.groupby("X11")["X9"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
        )
        x9_non_null = X["X9"].dropna()
        self.global_x9_mode_ = (
            x9_non_null.mode().iat[0] if not x9_non_null.mode().empty else "Medium"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["X1_prefix"] = X["X1"].str.slice(0, 2)

        x3_map = {
            "LF": "Low Fat",
            "low fat": "Low Fat",
            "Low Fat": "Low Fat",
            "reg": "Regular",
            "Regular": "Regular",
        }
        X["X3"] = X["X3"].map(x3_map).fillna(X["X3"])
        X.loc[X["X1_prefix"] == "NC", "X3"] = "Non-Edible"

        X.loc[X["X4"] == 0, "X4"] = np.nan
        X["X4"] = self._impute_by_group(
            X["X1"], X["X4"], self.x1_x4_mean_, self.global_x4_mean_
        )
        X["X2"] = self._impute_by_group(
            X["X1"], X["X2"], self.x1_x2_mean_, self.global_x2_mean_
        )

        X["Outlet_Years"] = OUTLET_REF_YEAR - X["X8"]

        x9_fill = X["X11"].map(self.x11_x9_mode_).fillna(self.global_x9_mode_)
        X["X9"] = X["X9"].where(X["X9"].notna(), x9_fill)

        x7_means = X["X7"].map(self.x7_x4_mean_).fillna(self.global_x4_imputed_mean_)
        X["X4_ratio"] = X["X4"] / x7_means.replace(0, np.nan)
        X["X4_ratio"] = X["X4_ratio"].fillna(1.0)

        X["X6_bin"] = pd.cut(
            X["X6"],
            bins=self.x6_bin_edges_,
            labels=["Q1", "Q2", "Q3", "Q4"],
            include_lowest=True,
        ).astype(object)
        X["X6_bin"] = X["X6_bin"].fillna("Q1")

        return X[ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL]

    @staticmethod
    def _impute_by_group(
        keys: pd.Series,
        values: pd.Series,
        group_means: pd.Series,
        global_mean: float,
    ) -> pd.Series:
        filler = keys.map(group_means)
        out = values.where(values.notna(), filler)
        return out.fillna(global_mean)

    def get_feature_names_out(self, input_features=None):
        return np.array(ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL)
