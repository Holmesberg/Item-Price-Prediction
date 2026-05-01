"""ColumnTransformer pipelines built on top of BigMartFeatures.

Three column groups feed into the model:
  - ENGINEERED_NUMERIC: numeric features (raw + engineered ratios).
  - ENGINEERED_CATEGORICAL: low-/medium-cardinality categoricals → OHE / Ordinal.
  - ENGINEERED_TARGET_ENCODE: high-cardinality categoricals (raw X1, 1553 unique)
    → sklearn TargetEncoder. The encoder uses internal 5-fold cross-fitting
    so each row's encoding is computed from folds *not containing* that row,
    avoiding within-fit leakage. Combined with the outer 5-fold CV's per-fold
    Pipeline refit (driven by GridSearchCV / cross_val_predict), the target
    encoding is leakage-safe at both levels.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
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


def _ohe(**kwargs):
    """OneHotEncoder shim: sklearn renamed sparse->sparse_output in 1.2."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)


def _target_encoder() -> TargetEncoder:
    return TargetEncoder(
        target_type="continuous",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=SEED,
    )


def build_linear_preprocessor() -> Pipeline:
    """For Ridge / KNN / NN: impute -> OHE+scale + target-encode+scale."""
    column_block = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                ENGINEERED_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _ohe()),
                    ]
                ),
                ENGINEERED_CATEGORICAL,
            ),
            (
                "te",
                Pipeline(
                    [
                        ("encode", _target_encoder()),
                        ("scale", StandardScaler()),
                    ]
                ),
                ENGINEERED_TARGET_ENCODE,
            ),
        ]
    )
    return Pipeline([("features", BigMartFeatures()), ("encode", column_block)])


def build_tree_preprocessor() -> Pipeline:
    """For RF / LightGBM / XGBoost: impute -> ordinal categoricals + target-encode."""
    column_block = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                ENGINEERED_NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                ENGINEERED_CATEGORICAL,
            ),
            (
                "te",
                _target_encoder(),
                ENGINEERED_TARGET_ENCODE,
            ),
        ]
    )
    return Pipeline([("features", BigMartFeatures()), ("encode", column_block)])
