"""ColumnTransformer pipelines built on top of BigMartFeatures."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import (
    BigMartFeatures,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
)


def _ohe(**kwargs):
    """OneHotEncoder shim: sklearn renamed sparse->sparse_output in 1.2."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)


def build_linear_preprocessor() -> Pipeline:
    """For Ridge / KNN: impute -> OneHot categoricals + StandardScale numerics."""
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
        ]
    )
    return Pipeline([("features", BigMartFeatures()), ("encode", column_block)])


def build_tree_preprocessor() -> Pipeline:
    """For RF / LightGBM: impute -> ordinal encode categoricals, no scaling."""
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
        ]
    )
    return Pipeline([("features", BigMartFeatures()), ("encode", column_block)])
