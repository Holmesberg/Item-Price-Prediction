"""Four model factories with different inductive biases.

Each returns a Pipeline of (preprocessor -> estimator). Hyperparameter grids
are exposed via param_grid() so train.py can tune them with a shared CV.
"""

from __future__ import annotations

from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    TargetEncoder,
)

from .config import SEED
from .features import (
    BigMartFeatures,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    ENGINEERED_TARGET_ENCODE,
)
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


def build_xgb(seed: int = SEED) -> Pipeline:
    from xgboost import XGBRegressor

    return Pipeline(
        [
            ("prep", build_tree_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    random_state=seed,
                    tree_method="hist",
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def _build_linear_with_poly() -> Pipeline:
    """Linear preprocessor with degree-2 interaction-only polynomial features
    on the numeric block. OHE/TE blocks unchanged. Used by ridge_poly only —
    other linear models (knn, nn) don't benefit from this expansion."""
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    te = TargetEncoder(
        target_type="continuous", smooth="auto", cv=5,
        shuffle=True, random_state=SEED,
    )
    column_block = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                    ]
                ),
                ENGINEERED_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", ohe),
                    ]
                ),
                ENGINEERED_CATEGORICAL,
            ),
            (
                "te",
                Pipeline([("encode", te), ("scale", StandardScaler())]),
                ENGINEERED_TARGET_ENCODE,
            ),
        ]
    )
    return Pipeline([("features", BigMartFeatures()), ("encode", column_block)])


def build_ridge_poly(seed: int = SEED) -> Pipeline:
    """Ridge with degree-2 interaction-only poly features on the numeric block.
    With 9 engineered numerics this adds 36 pair interactions, giving Ridge
    explicit access to e.g. logX6 × Outlet_Years, X1_count × X11_one_hot, etc."""
    return Pipeline(
        [
            ("prep", _build_linear_with_poly()),
            ("model", Ridge(alpha=10.0, random_state=seed)),
        ]
    )


def build_enet(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            ("model", ElasticNet(alpha=0.001, l1_ratio=0.9, random_state=seed, max_iter=50000, tol=1e-5)),
        ]
    )


def build_bridge(seed: int = SEED) -> Pipeline:
    # BayesianRidge auto-tunes alpha; doesn't take a random_state.
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            ("model", BayesianRidge(max_iter=300)),
        ]
    )


def build_huber(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            ("model", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000)),
        ]
    )


def build_bag_enet(seed: int = SEED) -> Pipeline:
    """Bagging ElasticNet — 15 bootstrap copies, averaged. Variance reduction
    on the new champion model."""
    base = ElasticNet(alpha=0.003, l1_ratio=0.85, random_state=seed,
                      max_iter=50000, tol=1e-5)
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            ("model", BaggingRegressor(
                estimator=base,
                n_estimators=15,
                max_samples=0.8,
                bootstrap=True,
                random_state=seed,
                n_jobs=-1,
            )),
        ]
    )


def build_bag_ridge(seed: int = SEED) -> Pipeline:
    """Bagging Ridge — 15 bootstrap copies, averaged."""
    base = Ridge(alpha=10.0, random_state=seed)
    return Pipeline(
        [
            ("prep", build_linear_preprocessor()),
            ("model", BaggingRegressor(
                estimator=base,
                n_estimators=15,
                max_samples=0.8,
                bootstrap=True,
                random_state=seed,
                n_jobs=-1,
            )),
        ]
    )


def build_hgbr(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_tree_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=500,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )


MODEL_REGISTRY: dict[str, Callable[[int], Pipeline]] = {
    "ridge": build_ridge,
    "ridge_poly": build_ridge_poly,
    "rf": build_rf,
    "lgbm": build_lgbm,
    "knn": build_knn,
    "nn": build_nn,
    "xgb": build_xgb,
    "enet": build_enet,
    "bridge": build_bridge,
    "huber": build_huber,
    "hgbr": build_hgbr,
    "bag_enet": build_bag_enet,
    "bag_ridge": build_bag_ridge,
}


def param_grid(name: str) -> dict[str, list]:
    """Per-model grids — fast on 6k rows. Keys are pipeline-style."""
    if name == "ridge":
        # Day-1 best was alpha=10; with target-encoded X1 added, the
        # optimum may shift either way. Widen the search both directions.
        return {"model__alpha": [0.1, 1.0, 10.0, 30.0, 100.0]}
    if name == "ridge_poly":
        # Polynomial features blow up dimensionality (~80+ feats), so
        # higher alphas are likely needed to control overfitting.
        return {"model__alpha": [1.0, 10.0, 30.0, 100.0, 300.0]}
    if name == "rf":
        return {
            "model__n_estimators": [500],
            "model__max_depth": [None, 12, 20],
            "model__min_samples_leaf": [1, 5],
        }
    if name == "lgbm":
        # Day-1 winner was 500@0.05 — clearly under-trained. Push to more
        # trees with lower lr (the standard well-tuned LGBM pattern on
        # small tabular). 16 combos × 5 folds.
        return {
            "model__n_estimators": [1000, 2500],
            "model__learning_rate": [0.02, 0.05],
            "model__num_leaves": [15, 31],
            "model__min_child_samples": [10, 30],
        }
    if name == "knn":
        return {
            "model__n_neighbors": [5, 10, 20, 50],
            "model__weights": ["uniform", "distance"],
        }
    if name == "nn":
        # Tight grid — NN training is the slowest model; 8 combos × 5 folds.
        return {
            "model__hidden": [128, 256],
            "model__dropout": [0.1, 0.3],
            "model__lr": [1e-3, 5e-4],
        }
    if name == "xgb":
        # 24 combos × 5 folds. xgb-hist is fast on 6k rows.
        return {
            "model__n_estimators": [500, 1000],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [4, 6, 8],
            "model__min_child_weight": [1, 5],
        }
    if name == "enet":
        # Iter 2: previous best 0.5206 at alpha=0.003 l1_ratio=0.85.
        # Refine around that point.
        return {
            "model__alpha": [0.002, 0.003, 0.005, 0.008, 0.012],
            "model__l1_ratio": [0.78, 0.85, 0.92],
        }
    if name == "bridge":
        # BayesianRidge auto-tunes; no grid (single fit).
        return {}
    if name == "huber":
        return {
            "model__alpha": [0.001, 0.01, 0.1],
            "model__epsilon": [1.1, 1.35, 1.5],
        }
    if name == "hgbr":
        # 12 combos × 5 folds — conservative, hgbr is the slowest new model.
        return {
            "model__max_iter": [500],
            "model__learning_rate": [0.03, 0.05],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__l2_regularization": [0.0, 1.0],
        }
    if name in ("bag_enet", "bag_ridge"):
        # Tiny grid: just sample fraction. 3 fits.
        return {"model__max_samples": [0.6, 0.8, 1.0]}
    raise KeyError(f"Unknown model: {name}")
