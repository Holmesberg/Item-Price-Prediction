"""Project-wide constants and seed control."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

SEED: int = 42

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
SAMPLE_SUB_CSV = ROOT / "sample_submission.csv"

RESULTS_DIR = ROOT / "results"
SUBMISSIONS_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"

# Y is already log-transformed (range ~3.51 to ~9.40). NEVER apply log1p again.
# RMSE on Y is assumed (== RMSLE on raw sales). Verify on the Kaggle Evaluation tab
# before finalizing submissions.
Y_LOG_MIN = 3.0
Y_LOG_MAX = 10.0

# Fixed reference year for Outlet_Years — keeps feature deterministic across re-runs.
OUTLET_REF_YEAR = 2013


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # PYTHONHASHSEED only takes effect at interpreter startup, not when set
    # at runtime. Reproducibility for sklearn/lightgbm/torch comes from the
    # explicit random_state= args wired through the model factories.
