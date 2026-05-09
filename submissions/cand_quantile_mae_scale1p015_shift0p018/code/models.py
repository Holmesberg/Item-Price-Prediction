"""Final model lineup.

The exploration phase trained many extra variants. The final repo keeps only
the five allowed/useful model families:

  - lasso_te200: tuned linear champion and current best public-LB submission
  - quantile_mae: median regression for the actual MAE leaderboard metric
  - rf: random forest baseline
  - lgbm: gradient-boosted trees
  - xgb: gradient-boosted trees from a second implementation

The tuned parameters are baked into the factories so `python -m src.train`
refits the final lineup instead of re-running the old model zoo.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, QuantileRegressor
from sklearn.pipeline import Pipeline

from .config import SEED
from .preprocessing import build_linear_preprocessor, build_tree_preprocessor


def build_lasso_te200(seed: int = SEED) -> Pipeline:
    """Best single-model LB candidate from the May 9 sweep."""
    return Pipeline(
        [
            ("prep", build_linear_preprocessor(te_smooth=200.0)),
            (
                "model",
                ElasticNet(
                    alpha=0.003,
                    l1_ratio=1.0,
                    random_state=seed,
                    max_iter=70000,
                    tol=1e-5,
                ),
            ),
        ]
    )


def build_quantile_mae(seed: int = SEED) -> Pipeline:
    """Median regression tuned for MAE instead of RMSE."""
    return Pipeline(
        [
            ("prep", build_linear_preprocessor(te_smooth=200.0)),
            (
                "model",
                QuantileRegressor(
                    quantile=0.5,
                    alpha=0.001,
                    solver="highs",
                ),
            ),
        ]
    )


def build_rf(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_tree_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_lgbm(seed: int = SEED) -> Pipeline:
    from lightgbm import LGBMRegressor

    return Pipeline(
        [
            ("prep", build_tree_preprocessor()),
            (
                "model",
                LGBMRegressor(
                    n_estimators=357,
                    learning_rate=0.015457424232386578,
                    num_leaves=18,
                    min_child_samples=33,
                    reg_alpha=3.738425892816665e-06,
                    reg_lambda=0.014762390709189041,
                    subsample=0.7721654847357287,
                    subsample_freq=1,
                    colsample_bytree=0.8478135772534396,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_xgb(seed: int = SEED) -> Pipeline:
    from xgboost import XGBRegressor

    return Pipeline(
        [
            ("prep", build_tree_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=232,
                    learning_rate=0.08285969532148683,
                    max_depth=10,
                    min_child_weight=35,
                    subsample=0.8598283564401406,
                    colsample_bytree=0.6898810916195833,
                    reg_alpha=2.688559321396769e-07,
                    reg_lambda=3.341467005810035e-08,
                    gamma=2.8334340919133427,
                    random_state=seed,
                    tree_method="hist",
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


MODEL_REGISTRY: dict[str, Callable[[int], Pipeline]] = {
    "lasso_te200": build_lasso_te200,
    "quantile_mae": build_quantile_mae,
    "rf": build_rf,
    "lgbm": build_lgbm,
    "xgb": build_xgb,
}


def param_grid(name: str) -> dict[str, list]:
    """No broad search by default: final tuned params live in the factories."""
    if name in MODEL_REGISTRY:
        return {}
    raise KeyError(f"Unknown model: {name}")
