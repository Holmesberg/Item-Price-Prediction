# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

CSE281 (Intro to AI) course project — solo entry into the Kaggle competition **`cse-281-spring-26-item-price-prediction`**. Tabular regression. Practical-exam deadline window is ~10 days from 2026-05-01.

The dataset is the well-known **BigMart Sales** dataset with anonymized columns. Knowing this unlocks domain-aware feature engineering that pure EDA won't suggest.

## Data

Files in repo root (no `data/` subdir):
- `train.csv` — 6,000 rows, columns `X1..X11, Y`
- `test.csv` — 2,523 rows, columns `X1..X11`
- `sample_submission.csv` — 2,523 rows, columns `row_id, Y`. **`row_id` is a 0-indexed integer matching test.csv row order**, not the BigMart `Item_Identifier_Outlet_Identifier` composite key. Preserve test row order when generating predictions.

### Critical non-obvious facts
- **`Y` is already log-transformed** (range 3.51–9.40, mean ~7.3, slight left skew). Do **not** re-apply `np.log1p`. Train, validate (RMSE), and submit directly in this space. The original BigMart `Item_Outlet_Sales` ranges ~33–13,000 — if you see numbers like that anywhere, something is wrong.
- **Evaluation metric is not yet confirmed** from the Kaggle competition's Evaluation tab. Assume RMSE on `Y` (== RMSLE on raw sales) until verified. If a future task depends on the metric, confirm first.

### Column mapping (BigMart → `Xn`)
| Col | Meaning | Notes |
|---|---|---|
| X1 | Item Identifier | 1553 unique. First 2 chars encode meta-category (`FD`=food, `DR`=drink, `NC`=non-consumable) — engineer as feature. |
| X2 | Item Weight | 16.8% missing in train, 18.1% in test. Missingness clusters by X1 → impute via groupby(X1) mean, then global fallback. |
| X3 | Item Fat Content | **Dirty: 5 values** (`LF`, `low fat`, `Low Fat`, `reg`, `Regular`). Collapse to {Low Fat, Regular}. Non-consumables (X1 prefix `NC`) should get a third category like "Non-Edible". |
| X4 | Item Visibility | **360 zeros are mislabeled missings** (a stocked product can't have 0% shelf visibility). Replace 0 → NaN, then impute (groupby X1 mean is a reasonable default). |
| X5 | Item Type | 16 categories. |
| X6 | Item MRP | Strongest single predictor by a wide margin. Binning into 4 quantile buckets is a known win. |
| X7 | Outlet Identifier | 10 unique. |
| X8 | Outlet Establishment Year | 1985–2009. Engineer `Outlet_Years = reference_year - X8` (use a fixed reference like 2013 for reproducibility, not `datetime.now`). |
| X9 | Outlet Size | 28.5% missing in train, 27.7% in test. Missingness clusters by X7 and correlates with X11 — impute via mode within X11 (Outlet Type). |
| X10 | Outlet Location Tier | 3 unique. |
| X11 | Outlet Type | 4 unique: `Grocery Store`, `Supermarket Type1/2/3`. |

## Course / competition constraints

From the project description doc:
- **≥4 different regression models** with genuinely different inductive biases (e.g., Linear/Ridge, Random Forest, XGBoost or LightGBM, plus one more like KNN/SVR/MLP).
- **Fixed random seed per model** (use one constant, e.g., `SEED = 42`, everywhere — splits, model init, samplers).
- **Save top-2 submissions** with the exact reproducible code that produced them.
- **PDF report** with EDA, preprocessing justification, model comparison table, feature importance, residual plots.
- **Only the provided dataset** — no external data.

Project 2's Kaggle page rules are placeholder text ("Don't cheat / Apply yourself / Have fun") — there is **no blackbox-model restriction** here (that rule is on Project 1, which we are not doing). XGBoost / LightGBM / CatBoost / sklearn are all fair game with no asterisks.

## Hardware

RTX A4000 (8GB VRAM), 32GB RAM. Irrelevant for tabular — boosted-tree training on 6k rows is seconds. Don't suggest GPU-heavy approaches.

## Repo layout

The repo currently only contains the data files and `Project Description.docx`. No source code, no `requirements.txt`, no notebooks, no git history yet — the user will scaffold the pipeline from scratch. Suggested structure when building it out:

```
notebooks/     # EDA, report visuals
src/           # preprocessing, model definitions, training loops
submissions/   # top-2 CSVs + the exact code that produced each
report/        # PDF + figures
```

## Working agreements

- When writing models, set the seed in **every** stochastic step (`train_test_split`, `RandomForestRegressor(random_state=...)`, `xgboost` `seed=`, `numpy`, `random`). One missed spot = non-reproducible top-2 submission.
- Validation strategy: a single fixed `train_test_split(..., random_state=SEED)` is fine for this scope; KFold is nicer for the report's model-comparison table but not required. Pick one and use it consistently across all 4 models so the comparison is honest.
- When generating a submission, write `row_id` as 0..N-1 in test.csv row order and `Y` in the same log space as training. Sanity-check that submission `Y` falls in roughly [3, 10] before saving.

## Session continuity

A comprehensive handoff document with full project state, model lineup,
candidate submissions, what worked / didn't, and open action items lives at
`HANDOFF.md` in this repo. Read it after this file when starting a new session.

## Diagnostics (already established — do not re-derive)

These were diagnosed during the modeling phase. Future sessions should treat them as facts and frame work around them, not re-investigate.

1. **Test-set X1 overlap with train: 99.3%** (only 17/2523 unseen items).
   The problem is dominated by per-item mean estimation, not feature
   engineering or model architecture. Any reasonable regularizer solves it.

2. **All reasonable models converge.** OOF correlation matrix across
   13 trained models (Ridge, ENet, Bridge, Huber, LGBM, XGB, RF, HGBR,
   CatBoost, KNN, NN, RidgePoly, LGBM-MAE) shows >0.98 pairwise
   correlation, with most >0.99. Adding more models is futile.

3. **CV-LB gap is reverse-direction (CV ~0.52, LB ~0.379).** Not
   leakage. CV folds randomly distribute rare-X1 rows into validation;
   test set is concentrated on well-represented items. The hard cases
   are over-represented in CV, under-represented in LB.

4. **Leaderboard top-3 tied at 0.371.** Reflects the noise floor
   imposed by (1). Realistic ceiling for any submission is ~0.371-0.375.

Implication: pick final submissions by **LB score, not CV**. The CV
ranking is informative but optimistically biased toward complex models
that handle rare X1 well — irrelevant for the test distribution.
