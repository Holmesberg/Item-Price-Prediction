"""Four model factories with different inductive biases.

Each returns a Pipeline of (preprocessor -> estimator). Hyperparameter grids
are exposed via param_grid() so train.py can tune them with a shared CV.
"""

from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from .config import SEED
from .nn_model import SklearnNNRegressor
from .preprocessing import build_linear_preprocessor, build_tree_preprocessor


def build_ridge(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            ("model", Ridge(alpha=1.0, random_state=seed)),
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
                    max_depth=None,
                    min_samples_leaf=1,
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
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_knn(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            (
                "model",
                KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1),
            ),
        ]
    )


def build_nn(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            (
                "model",
                SklearnNNRegressor(
                    hidden=128,
                    dropout=0.2,
                    lr=1e-3,
                    batch_size=64,
                    epochs=30,
                    early_stopping=True,
                    patience=5,
                    seed=seed,
                ),
            ),
        ]
    )


MODEL_REGISTRY: dict[str, Callable[[int], Pipeline]] = {
    "ridge": build_ridge,
    "rf": build_rf,
    "lgbm": build_lgbm,
    "knn": build_knn,
    "nn": build_nn,
}


def param_grid(name: str) -> dict[str, list]:
    """Small per-model grids — fast on 6k rows. Keys are pipeline-style."""
    if name == "ridge":
        return {"model__alpha": [0.1, 1.0, 10.0]}
    if name == "rf":
        return {
            "model__n_estimators": [500],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 5],
        }
    if name == "lgbm":
        return {
            "model__n_estimators": [500, 1000],
            "model__learning_rate": [0.05, 0.1],
            "model__num_leaves": [31, 63],
            "model__min_child_samples": [5, 20],
        }
    if name == "knn":
        return {
            "model__n_neighbors": [5, 10, 20, 50],
            "model__weights": ["uniform", "distance"],
        }
    if name == "nn":
        # Tight grid — NN training is the slowest model; 12 combos × 5 folds.
        return {
            "model__hidden": [128, 256],
            "model__dropout": [0.1, 0.3],
            "model__lr": [1e-3, 5e-4],
        }
    raise KeyError(f"Unknown model: {name}")
