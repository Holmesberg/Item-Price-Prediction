# CSE281 Item Price Prediction: Discussion Guide

This document is the end-to-end explanation of the repository as of May 9,
2026. Use it to prepare for a project discussion, viva, practical exam, or
code walkthrough.

## How To Use This Guide

If you are already comfortable with machine learning competitions, skim the
first few sections and focus on the model lineup, leakage control, and
leaderboard calibration.

If you are newer to ML, read in this order:

1. Read **Prerequisite Knowledge**.
2. Read **Glossary Of Terms**.
3. Read **The Project In One Picture**.
4. Read **Problem Summary**.
5. Read **Feature Engineering**, **Preprocessing Pipelines**, and **Final Model
   Lineup**.
6. End with **What To Say In A Presentation** and **Questions You May Be
   Asked**.

The main thing to understand is not every code detail. The main thing is the
story:

> This is a tabular prediction problem where item identity is the strongest
> signal. The pipeline safely encodes item information, compares a small set of
> models, and uses a calibrated lasso submission as the best public-LB result.

## Prerequisite Knowledge

You should be comfortable with these ideas before discussing the project.

### 1. Supervised Learning

Supervised learning means the training data has examples with known answers.

In this project:

- Input features: columns like `X1`, `X2`, `X3`, ..., `X11`.
- Target/label: `Y`.
- Goal: learn a function that maps input features to `Y`.

Plain-English version:

> We show the model many rows where the answer is known. Later, we ask it to
> predict the answer for rows where the answer is hidden.

### 2. Regression

Regression means predicting a number.

This project is regression because `Y` is numeric. It is not classification
because there are no classes like "yes/no" or "category A/B/C".

### 3. Tabular Data

Tabular data is spreadsheet-like data:

```text
row | X1    | X2  | X3      | ... | Y
0   | FDA15 | 9.3 | Low Fat | ... | 8.23
1   | DRC01 | 5.9 | Regular | ... | 6.09
```

Each row is one item/outlet example. Each column is a property of that example.

### 4. Train/Test Split

`train.csv` has rows where `Y` is known. The model learns from these rows.

`test.csv` has rows where `Y` is hidden. The model predicts these rows and
writes a Kaggle submission.

### 5. Validation

Validation means pretending some training rows are "hidden" so we can test the
model locally.

We cannot see the true `Y` for Kaggle test rows, so validation gives us an
estimate of how well the model might perform.

### 6. Overfitting

Overfitting means the model memorizes training data too closely and performs
badly on new data.

Example:

- Good: "Items of this type in this outlet usually sell around this range."
- Bad: "This exact training row had this exact target, so I will memorize it."

Target encoding and leaderboard calibration both need careful explanation
because they can overfit if done incorrectly.

### 7. Leakage

Leakage means accidentally using information during training that would not be
available at prediction time.

The dangerous version:

> Computing a target-encoded value for a row using that same row's `Y`.

The safe version:

> Computing encodings inside CV folds, so validation rows are encoded using
> only the training part of that fold.

Leakage is one of the most important things your instructor may ask about.

## Glossary Of Terms

### Feature

A feature is an input column used by the model.

Examples:

- `X1`: item ID.
- `X6`: price/MRP-like value.
- `Outlet_Years`: engineered outlet age feature.

### Target / Label

The target is the value we want to predict. Here it is `Y`.

### Prediction

A prediction is the model's estimated `Y` for a row.

### Model

A model is the algorithm that learns patterns from the training data.

Examples in this repo:

- Lasso/ElasticNet.
- Quantile regression.
- Random forest.
- LightGBM.
- XGBoost.

### Pipeline

An sklearn `Pipeline` chains steps together.

In this project, a pipeline usually means:

```text
raw data -> feature engineering -> encoding/scaling -> model -> prediction
```

Why pipelines matter:

- They keep preprocessing and modeling together.
- They make CV safer.
- They reduce accidental leakage.

### Cross-Validation (CV)

Cross-validation splits the training data into folds. The model trains on some
folds and validates on the remaining fold.

This repo uses 5-fold CV:

```text
Fold 1: validate on part 1, train on parts 2-5
Fold 2: validate on part 2, train on parts 1,3,4,5
...
Fold 5: validate on part 5, train on parts 1-4
```

### OOF Predictions

OOF means out-of-fold.

An OOF prediction for a training row is made by a model that did not train on
that row. This gives a more honest local estimate than predicting on the same
data used for training.

### MAE

MAE means Mean Absolute Error.

Formula:

```text
MAE = average(|actual Y - predicted Y|)
```

If actual `Y = 8.0` and predicted `Y = 7.7`, the absolute error is `0.3`.

Kaggle uses MAE for this competition.

### RMSE

RMSE means Root Mean Squared Error.

Formula:

```text
RMSE = sqrt(average((actual Y - predicted Y)^2))
```

RMSE punishes large errors more strongly than MAE. Earlier project work used a
lot of RMSE diagnostics, but the final competition metric is MAE.

### Public Leaderboard (Public LB)

Kaggle reveals a score on part of the hidden test set. That is the public
leaderboard score.

Important caveat:

> A better public score does not always mean better true generalization,
> because it is only one slice of the test data.

### Private Leaderboard

Some competitions reveal a final private score after the deadline. If this
competition uses a hidden/private split, public-LB calibration may not transfer
perfectly.

### Target Encoding

Target encoding replaces a category with a target statistic.

Example:

If item `FDA15` usually has target around `8.2`, target encoding gives `FDA15`
a numeric value near `8.2`.

Why it is useful:

- `X1` has many item IDs.
- One-hot encoding every item can be sparse/noisy.
- Target encoding gives the model a compact item-level signal.

Why it is dangerous:

- If done incorrectly, it can leak the target.

How this repo handles it:

- Target encoding is inside sklearn pipelines.
- It uses cross-fitting.
- Validation rows do not encode themselves using their own target.

### Regularization

Regularization is a penalty that keeps a model from fitting noise too strongly.

Plain-English version:

> Regularization tells the model: learn the pattern, but do not trust every tiny
> fluctuation in the training data.

Lasso and ElasticNet are regularized linear models.

### Lasso

Lasso is a linear regression model with L1 regularization.

Why useful here:

- It works well with many encoded features.
- It can shrink less useful feature weights toward zero.
- It is stable on small-to-medium tabular data.

### Quantile Regression

Quantile regression predicts a quantile of the target distribution.

This repo uses `quantile=0.5`, which predicts the median.

Why this matters:

- The median is naturally connected to MAE.
- For absolute error, predicting the median can be better than predicting the
  mean.

### Random Forest

A random forest is many decision trees averaged together.

It is useful as a nonlinear baseline, but here it did not beat the linear
models.

### Gradient Boosting

Gradient boosting builds trees sequentially, where each new tree tries to fix
errors from previous trees.

LightGBM and XGBoost are gradient boosting libraries.

## The Project In One Picture

Here is the whole project as a data flow:

```text
train.csv + test.csv
        |
        v
BigMartFeatures
  - clean categories
  - impute missing values
  - create counts and outlet/item features
        |
        v
Preprocessing
  - scale numeric features for linear models
  - encode categorical features
  - target-encode high-cardinality features safely
        |
        v
Models
  - lasso_te200
  - quantile_mae
  - rf
  - lgbm
  - xgb
        |
        v
Validation
  - 5-fold CV
  - OOF MAE/RMSE
        |
        v
Submission
  - fit selected model on all train rows
  - predict all test rows
  - write submission.csv
        |
        v
Leaderboard calibration
  - small scale/shift transform on lasso predictions
  - best public score: 0.370
```

Beginner translation:

> The code cleans the spreadsheet, turns text categories into numbers, trains a
> few models, checks them locally, writes Kaggle files, and finally applies a
> small adjustment to the best lasso predictions for the public leaderboard.

## Short Answer: Did We Add New Models?

Yes, but only one new active model was added to the final <=5 model lineup:

- `quantile_mae`: a `QuantileRegressor` median model, added because Kaggle
  evaluates Mean Absolute Error (MAE), not RMSE.

The active final registry in `src/models.py` now contains exactly five models:

1. `lasso_te200`
2. `quantile_mae`
3. `rf`
4. `lgbm`
5. `xgb`

The old `nn` model was removed from the active registry to keep the final
lineup within the 5-model rule. Its wrapper still exists in `src/nn_model.py`
because historical submission snapshots and old artifacts may refer to it, but
it is not in `MODEL_REGISTRY`.

There is also a Huber probe in the submissions/results directories. That was
not added as a final active model. It was a legacy saved-model experiment used
to test whether a robust linear direction helped public MAE. It scored worse
on the public leaderboard and should be described as a negative probe, not part
of the final lineup.

## Current Best Submission

Best confirmed public leaderboard score: `0.370`.

The recommended final public-LB candidate is:

```text
submissions/cand_lasso_scale1p05_shift0p08/submission.csv
```

Equivalent public score ties were also seen from nearby lasso calibration
points:

- `cand_lasso_scale1p05_shift0p08`: public `0.370`
- `cand_lasso_scale1p0664144141426704_shift0p07278889909376063`: public `0.370`
- `cand_lasso_scale1p07_shift0p08`: public `0.370`

The best model family is still the calibrated lasso-style linear model, even
though `quantile_mae` has the best local OOF MAE.

## Problem Summary

The task is a BigMart-style item price/sales prediction problem. The input is
tabular retail data with product, outlet, visibility, weight, and price-like
features. The target column is `Y`.

Beginner explanation:

> Each row describes an item being sold at an outlet. We do not know the real
> target for the test rows, so we train on historical rows and predict the
> missing `Y` values.

The columns are anonymized as `X1`, `X2`, etc., but they correspond closely to
the well-known BigMart sales dataset. You do not need to memorize the exact
business meaning of every column, but you should understand what kind of
information each one carries.

| Column | Type | Practical meaning in this project |
|---|---|---|
| `X1` | categorical | Item ID. This is the strongest signal. |
| `X2` | numeric | Item weight-like value. Missing values are imputed by item. |
| `X3` | categorical | Fat-content-like category, cleaned into consistent labels. |
| `X4` | numeric | Item visibility-like value. Zeros are treated as missing. |
| `X5` | categorical | Item type/category. |
| `X6` | numeric | Price/MRP-like value. Very important for sales level. |
| `X7` | categorical | Outlet/store ID. |
| `X8` | numeric/year | Outlet establishment year. Converted to outlet age. |
| `X9` | categorical | Outlet size. Missing values are imputed. |
| `X10` | categorical | Outlet location tier. |
| `X11` | categorical | Outlet type. |
| `Y` | numeric target | The value we predict. Already log-transformed. |

Important target fact:

- `Y` is already log-transformed.
- Do not apply `log1p`, `exp`, or reverse transforms before submission.
- Kaggle evaluates MAE directly on `Y`.

Why the log-transformed target matters:

If the original sales value was very skewed, taking a log makes it easier for
models to learn. But because the competition already gives us `Y` on the log
scale, applying another log would damage the target. Our predictions must stay
on the same scale as `Y`.

Data files:

- `train.csv`: 6000 labeled rows.
- `test.csv`: 2523 unlabeled rows.
- `sample_submission.csv`: expected submission shape.

Definitions:

- Labeled row: a row where we know `Y`.
- Unlabeled row: a row where `Y` is hidden and must be predicted.
- Submission row: a row in `submission.csv` containing a `row_id` and predicted
  `Y`.

Submission format:

```text
row_id,Y
0,<prediction for first test row>
1,<prediction for second test row>
...
```

The project predicts `Y` for all 2523 test rows.

## What The Model Is Actually Learning

The model is trying to answer questions like:

- Does this item usually sell at a high or low target value?
- Does this outlet type tend to produce higher targets?
- Does the price/MRP-like feature `X6` suggest a larger target?
- Is this item's visibility unusually high or low?
- How old is the outlet?
- Are there combinations of item type and outlet type that matter?

The most important lesson from exploration was:

> Item identity `X1` explains a huge amount of the target behavior.

That changes the modeling strategy. Instead of needing a very complex neural
network, we mainly need a safe way to summarize item-level information and
regularize it so it does not overfit.

## High-Level Repository Map

Root files:

- `train.csv`, `test.csv`, `sample_submission.csv`: competition data.
- `requirements.txt`: Python dependencies.
- `PROJECT_DISCUSSION_GUIDE.md`: this explanation.
- `HANDOFF.md`, `CLAUDE.md`: historical working notes. These may contain stale
  or older-session information; use this guide plus current source files for
  the final state.

Core code in `src/`:

- `config.py`: paths, seed, target range, output directories.
- `features.py`: domain feature engineering transformer.
- `preprocessing.py`: sklearn preprocessing pipelines.
- `models.py`: final <=5 model registry.
- `train.py`: trains registered models with 5-fold CV.
- `final_lb_candidates.py`: regenerates the original lasso candidate.
- `lb_calibration.py`: generates affine-calibrated lasso leaderboard probes.
- `quantile_mae_candidate.py`: trains/writes the MAE quantile candidate.
- `joblib_calibration.py`: writes calibrated submissions from saved legacy
  joblib models.
- `submit.py`: wrapper to run the final candidate script.
- `eda.py`, `plots.py`, `render_report.py`: reporting and visualization tools.

Outputs:

- `results/model_comparison.csv`: compact comparison of final models.
- `results/oof_*.npy`: ignored local OOF arrays for model diagnostics.
- `results/best_*.joblib`: ignored local fitted model artifacts.
- `results/cand_*.json`: tracked summaries for generated candidates.
- `submissions/<candidate>/submission.csv`: Kaggle submission files.
- `submissions/<candidate>/code/`: frozen source snapshot for reproducibility.

## The Main Idea

The strongest signal is the item identity `X1`.

Most test items also appear in training. That means the problem is dominated by
estimating product-level conditional averages, not by discovering complex
nonlinear structure. This is why regularized linear models with target encoding
perform very well, and why adding many different models gives tiny returns.

The practical narrative:

1. Clean and normalize the tabular fields.
2. Create item/outlet-aware features.
3. Use leakage-safe target encoding for high-cardinality item IDs.
4. Fit regularized linear models and a few tree baselines.
5. Diagnose that public test is easier than random CV because item overlap is
   high.
6. Apply small public-LB calibration to the best lasso prediction vector.

## Feature Engineering

All feature engineering lives in `src/features.py`, inside `BigMartFeatures`.
It is an sklearn-compatible transformer, so it is fit only on the training fold
during CV.

Definition:

> Feature engineering means creating better input columns from the raw data.

Why we need it:

Raw columns are often messy or not directly ideal for a model. For example,
`X8` is a year, but the model usually benefits more from outlet age. So the
code converts `X8` into `Outlet_Years = 2013 - X8`.

The feature engineering here has four goals:

1. Clean inconsistent text categories.
2. Fill missing or suspicious values.
3. Create domain-aware numeric signals.
4. Preserve important item/outlet identity information safely.

Raw numeric columns:

- `X2`
- `X4`
- `X6`
- `X8`

Raw categorical columns:

- `X1`
- `X3`
- `X5`
- `X7`
- `X9`
- `X10`
- `X11`

Engineered numeric features:

- `X2`: item weight-like feature, group-imputed by item `X1`.
- `X4`: visibility-like feature, zeros treated as missing then imputed.
- `X6`: price/MRP-like feature.
- `Outlet_Years`: `2013 - X8`.
- `X4_ratio`: visibility relative to outlet average visibility.
- `X1_count`: frequency of the item ID.
- `X7_count`: frequency of the outlet ID.
- `X1_X7_count`: frequency of the item-outlet pair.
- `logX6`: `log1p(X6)`, useful because sales relationships are often
  multiplicative.
- `X4_dev_X1`: visibility deviation from the item-level pooled visibility mean.

Engineered categorical features:

- `X1_prefix`: first two characters of `X1`, usually item class.
- `X3`: normalized fat-content category.
- `X5`: item type.
- `X6_bin`: quartile bin of `X6`.
- `X7`: outlet ID.
- `X9`: outlet size, mode-imputed within outlet type.
- `X10`: outlet location tier.
- `X11`: outlet type.

Target-encoded features:

- `X1`: high-cardinality item ID.
- `X5_X11`: item type by outlet type interaction.

### Step-By-Step Feature Engineering

This is what `BigMartFeatures` does in plain English.

### `X1_prefix`

The code takes the first two characters of `X1`.

Example:

```text
FDA15 -> FD
DRC01 -> DR
NCX06 -> NC
```

Why:

- The prefix groups items into broad classes.
- This gives the model a lower-cardinality item category.

### Cleaning `X3`

`X3` contains inconsistent labels, such as:

- `LF`
- `low fat`
- `Low Fat`
- `reg`
- `Regular`

The code maps them into consistent categories.

Why:

If we do not clean these, the model may think `LF` and `Low Fat` are different
categories even though they mean the same thing.

### Handling `X4` Visibility

The code treats `X4 == 0` as missing.

Why:

In retail visibility data, a true visibility of exactly zero is suspicious.
It often means missing or unrecorded visibility rather than "the item was
literally impossible to see."

The code fills missing `X4` values using the item's average visibility when
possible, then falls back to the global mean.

### Handling Missing `X2`

`X2` is imputed by item ID `X1`.

Why:

If the same item appears multiple times, its weight-like value should usually
be stable. Item-level imputation is more informative than simply using the
global average.

### `Outlet_Years`

The code creates:

```text
Outlet_Years = 2013 - X8
```

Why:

Models usually understand "age" more directly than "year established."

### Count Features

The code creates:

- `X1_count`
- `X7_count`
- `X1_X7_count`

Definition:

> A count feature tells the model how frequently something appears.

Why count features help:

- Common items have more evidence and may be easier to predict.
- Rare items may need stronger regularization.
- Item-outlet frequency can indicate how familiar the training data is with
  that exact combination.

### `logX6`

The code creates:

```text
logX6 = log1p(X6)
```

Important distinction:

- We do not log-transform `Y` again.
- We do create a log feature from `X6`.

Why:

Price/sales relationships are often multiplicative. A log-transformed price
feature can make that relationship easier for a linear model to capture.

### `X4_dev_X1`

This means:

```text
current visibility - average visibility for this item
```

Why:

The raw visibility value may be less informative than whether the item is more
or less visible than usual for itself.

### `X5_X11`

The code creates an interaction:

```text
item type + "_" + outlet type
```

Example:

```text
Dairy_Supermarket Type1
```

Why:

An item type may behave differently depending on outlet type. This feature
captures that combined effect.

## Leakage Control

This is one of the most important topics to explain.

The code avoids target leakage by putting feature engineering and encoding
inside sklearn `Pipeline` objects. During `cross_val_predict`, each fold fits
its own preprocessing pipeline on only that fold's training data.

Target encoding is handled with sklearn's `TargetEncoder`:

- It uses internal cross-fitting.
- A row's target encoding is not computed from its own target value.
- The outer CV split also refits the entire pipeline per fold.

The project does pool `test.csv` into count statistics through
`BigMartFeatures.EXTRA_COUNT_REF`, but only for target-independent counts and
visibility means:

- `X1_count`
- `X7_count`
- `X1_X7_count`
- pooled item visibility mean used by `X4_dev_X1`

This is allowed unsupervised use of test features. It does not read `Y` from
test because test has no target labels.

If asked whether this is leakage, say:

> We use test features only for unsupervised frequency/count statistics. We do
> not use test labels. Target-dependent transforms are fit inside CV folds, and
> sklearn's TargetEncoder cross-fits internally.

## Preprocessing Pipelines

`src/preprocessing.py` builds two preprocessors.

### Linear Preprocessor

Used by:

- `lasso_te200`
- `quantile_mae`

Steps:

1. `BigMartFeatures`
2. Numeric block:
   - median imputation
   - standard scaling
3. Categorical block:
   - most-frequent imputation
   - one-hot encoding
4. Target-encoding block:
   - sklearn `TargetEncoder`
   - standard scaling

Why this works:

- Linear models need scaled numeric features.
- Low-cardinality categoricals are safe to one-hot encode.
- High-cardinality `X1` is better target-encoded than one-hot encoded.

### Tree Preprocessor

Used by:

- `rf`
- `lgbm`
- `xgb`

Steps:

1. `BigMartFeatures`
2. Numeric block:
   - median imputation
3. Categorical block:
   - most-frequent imputation
   - ordinal encoding
4. Target-encoding block:
   - sklearn `TargetEncoder`

Why this differs:

- Tree models do not require feature scaling.
- Ordinal encoding is more compact for tree models than dense one-hot
  encoding.

## Final Model Lineup

The active registry in `src/models.py` has five models.

### 1. `lasso_te200`

Implementation:

- `ElasticNet(alpha=0.003, l1_ratio=1.0)`
- Because `l1_ratio=1.0`, this is effectively Lasso.
- Uses `TargetEncoder(smooth=200.0)`.

Role:

- Best practical public-LB model after calibration.
- Simple, regularized, and well suited to high-cardinality item signal.

Why it works:

- The dataset is dominated by item effects.
- Target-encoded item ID plus regularized linear regression gives strong,
  stable predictions.

Beginner explanation:

> Lasso is like drawing a straight-line formula using many encoded features,
> while also penalizing unnecessary feature weights. It is simple, but in this
> dataset simple is powerful because the engineered item features already carry
> the key signal.

### 2. `quantile_mae`

Implementation:

- `QuantileRegressor(quantile=0.5, alpha=0.001, solver="highs")`
- Uses the same linear preprocessor and `TargetEncoder(smooth=200.0)`.

Why it was added:

- Kaggle's metric is MAE.
- The median minimizes absolute error, so quantile regression is a natural
  metric-aligned model.

Result:

- Best local OOF MAE in `results/model_comparison.csv`: `0.39627`.
- Public LB was `0.371`, worse than calibrated lasso's `0.370`.

How to explain this:

> Quantile regression optimized the local validation metric, but the public
> test distribution favored the calibrated lasso prediction vector. It was a
> reasonable metric-driven addition, but not the final public-LB winner.

Beginner explanation:

> If MAE cares about absolute distance, predicting a median is often sensible.
> That is why this model was worth adding. It made the local validation table
> look better, but Kaggle's public split still liked the calibrated lasso more.

### 3. `rf`

Implementation:

- `RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5)`

Role:

- Nonlinear bagged-tree baseline.

Result:

- Worse local OOF MAE than the linear models.

Beginner explanation:

> A random forest averages many decision trees. It can model nonlinear rules,
> but here it did not beat the cleaner item-encoded linear approach.

### 4. `lgbm`

Implementation:

- Tuned `LGBMRegressor`
- Small leaves, low learning rate, regularization.

Role:

- Gradient boosting baseline.

Result:

- Useful as a sanity check, but not better than lasso/quantile.

Beginner explanation:

> LightGBM is a strong tree-boosting algorithm. It often wins tabular
> competitions, but this dataset's main signal was already captured by item
> target encoding, so boosting did not add enough extra value.

### 5. `xgb`

Implementation:

- Tuned `XGBRegressor`
- Histogram tree method.

Role:

- Second gradient boosting implementation for comparison.

Result:

- Better than RF/LGBM locally, but still not better than the linear family.

Beginner explanation:

> XGBoost is another strong boosted-tree method. It was useful for comparison,
> but it also made predictions very similar to the linear models and did not
> beat calibrated lasso.

## Current Local Model Comparison

From `results/model_comparison.csv`:

| Model | OOF MAE | OOF RMSE | Notes |
|---|---:|---:|---|
| `quantile_mae` | 0.39627 | 0.52504 | Best local MAE, public 0.371 |
| `lasso_te200` | 0.40152 | 0.52041 | Best public after calibration |
| `xgb` | 0.40593 | 0.52442 | Strongest tree baseline |
| `lgbm` | 0.40813 | 0.52709 | Tuned boosting baseline |
| `rf` | 0.41429 | 0.53351 | Tree baseline |

How to read this table:

- Lower MAE is better because Kaggle evaluates MAE.
- `OOF MAE` estimates validation performance using out-of-fold predictions.
- `OOF RMSE` is kept as an additional diagnostic because large errors still
  matter, even though it is not the official metric.
- `fit_seconds` tells you roughly how expensive the model is to train.
- `best_params` lists tuned hyperparameters, or `{}` when the factory already
  has fixed parameters.

Beginner note:

> Do not say "quantile is the final best model" just because it has the best
> OOF MAE. The final leaderboard winner is the calibrated lasso. The correct
> statement is: quantile was best locally, calibrated lasso was best publicly.

Why OOF MAE and public LB disagree:

- Public LB is a specific subset of test rows.
- The public subset appears easier than random CV folds.
- Calibration can improve public LB while worsening local OOF.
- This is useful for leaderboard placement, but should be described honestly as
  public-leaderboard calibration, not as proof of better generalization.

### Why Validation And Leaderboard Can Disagree

This is a subtle but important point.

In CV, the model trains on only 80 percent of the training rows in each fold.
For some items, that means the model sees fewer examples of that item than it
would see when trained on the full dataset.

On Kaggle test rows, many item IDs also exist in the full training set. So when
we fit on all 6000 training rows before submission, the model has stronger
item-level evidence than it had in each CV fold.

That can create this pattern:

```text
local CV looks worse
public leaderboard looks better
```

The key explanation:

> CV is pessimistic because each fold hides some item evidence. Full-train
> submission sees more item history, and the test split has high item overlap.

## Leaderboard Calibration

`src/lb_calibration.py` applies an affine transform to lasso predictions:

```text
prediction = mean(test_pred) + scale * (test_pred - mean(test_pred)) + shift
```

Meaning:

- `scale` changes prediction spread.
- `shift` moves all predictions up/down.
- Centering around the test mean keeps the transform interpretable.

Important submitted probes:

| Candidate | Public LB | Lesson |
|---|---:|---|
| raw `lasso_te200` | 0.378 | Strong base model |
| `scale=1.08, shift=0` | 0.380 | Too much spread alone hurts |
| `scale=1.00, shift=0.05` | 0.374 | Public slice wanted upward shift |
| `scale=1.0276, shift=0.0551` | 0.371 | Combining shift/spread helped |
| `scale=1.1054, shift=0.1114` | 0.372 | Too aggressive |
| `scale=1.05, shift=0.08` | 0.370 | Best confirmed region |
| `scale=1.0664, shift=0.0728` | 0.370 | Same rounded score |
| `scale=1.07, shift=0.08` | 0.370 | Same rounded score |

How to explain this carefully:

> Once we realized the metric was MAE and the public test split was easier than
> random CV, we used a few low-dimensional calibration probes on the best
> prediction vector. This is not a new model family; it is post-processing of
> the lasso predictions. It improved public LB from 0.378 to 0.370, but it may
> not transfer perfectly to a private split.

## Why Linear Models Beat Trees Here

The data is not huge: 6000 labeled rows.

The most important predictor is item identity `X1`, which has many categories.
Once `X1` is target-encoded safely, the model mostly needs to regularize noisy
item means and combine them with outlet/price features.

Linear models are good here because:

- They regularize high-cardinality item signal cleanly.
- They avoid overfitting small groups.
- They are stable across folds.
- The engineered features already encode the main nonlinear/domain structure.

Tree models are less competitive because:

- They need more data to reliably learn high-cardinality interactions.
- Ordinal-encoded categories can be awkward.
- Boosting can fit noise in rare item/outlet combinations.

## Important Negative Results

These are valuable to mention because they show disciplined experimentation.

### Huber Probe

Why tested:

- Huber loss is robust to outliers and had decent old OOF MAE.

Result:

- Public LB `0.375`, worse than calibrated lasso.

Conclusion:

- Robust loss did not solve the public error pattern.

### Quantile Regression

Why tested:

- MAE is the official metric, so median regression is theoretically aligned.

Result:

- Best local OOF MAE.
- Public LB `0.371`, not better than calibrated lasso.

Conclusion:

- Metric alignment helped validation but not the exposed public split.

### Aggressive Calibration

Why tested:

- Early public points suggested a larger scale/shift might reach the top score.

Result:

- `scale=1.1054, shift=0.1114` scored `0.372`, worse than the smaller
  calibration.

Conclusion:

- Public-LB calibration has a narrow useful basin; aggressive extrapolation
  overfits.

### Extra Model Zoo / Stacking

Earlier work trained many additional models and blends. Their OOF predictions
were highly correlated, so stacking gave very small gains. The final repo was
pruned back to a <=5 model lineup.

Conclusion:

- More models were mostly redundant.
- The bottleneck is item-mean estimation, not architecture variety.

## How Training Works In The Code

The main training script is `src/train.py`.

Beginner-level walkthrough:

1. Load `train.csv`.
2. Split `Y` away from the input columns.
3. Load `test.csv` so its feature columns can be used for unsupervised count
   statistics.
4. Create a 5-fold CV splitter.
5. For each model in `MODEL_REGISTRY`:
   - build the sklearn pipeline;
   - fit it inside CV;
   - generate OOF predictions;
   - compute OOF MAE and OOF RMSE;
   - save OOF predictions to `results/oof_<model>.npy`;
   - save the fitted model to `results/best_<model>.joblib`;
   - update `results/model_comparison.csv`.

In simple terms:

> `train.py` asks every final model: "How well do you predict rows you did not
> train on?" Then it stores the answers in the results folder.

### Why We Save OOF Predictions

OOF predictions are useful for:

- comparing models fairly;
- analyzing residuals;
- checking whether models make similar mistakes;
- trying blends without retraining everything.

Earlier in the project, OOF predictions showed that most models were highly
correlated. That was one reason to remove unnecessary model complexity.

### Why We Save Joblib Models

`joblib` is a common way to save sklearn models.

Saving `best_<model>.joblib` means we can reload a trained pipeline later and
predict test rows without retraining.

The repo ignores these files in git because they are large generated artifacts.

## How A Submission Is Created

A Kaggle submission is just a CSV with two columns:

```text
row_id,Y
```

For this project, a submission script usually does this:

1. Load `train.csv` and `test.csv`.
2. Fit the chosen model on all labeled training rows.
3. Predict `Y` for all test rows.
4. Optionally apply a calibration transform.
5. Write `submissions/<candidate>/submission.csv`.
6. Copy a snapshot of `src/` into `submissions/<candidate>/code/`.
7. Write a `note.txt` explaining the candidate.

The code snapshot is important because it lets us reconstruct exactly what code
created a submission.

## The Difference Between Model Improvement And Calibration

This project has both model work and calibration work.

Model improvement means changing the learning algorithm or features.

Examples:

- adding `quantile_mae`;
- tuning XGBoost;
- changing target encoding smoothness;
- adding feature engineering.

Calibration means taking predictions from an existing model and adjusting them.

Example:

```text
prediction = mean + scale * (prediction - mean) + shift
```

The best public score came from calibration of the lasso predictions. This is
not the same as discovering a new model; it is post-processing.

How to say it:

> The base lasso model provided the best prediction direction. The final public
> score improved after a small scale/shift calibration of that direction.

Why to be careful:

> Calibration is tuned using public leaderboard feedback, so it can overfit the
> public split. It is useful for leaderboard rank, but it should be reported
> separately from local validation results.

## Reproducibility Commands

Use WSL Python if Windows Python lacks the required packages.

Train all active final models:

```bash
python3 -m src.train
```

Train only quantile MAE:

```bash
python3 -m src.train quantile_mae
```

Regenerate the original lasso candidate:

```bash
python3 -m src.final_lb_candidates
```

Generate the best-style calibrated lasso candidate:

```bash
python3 -m src.lb_calibration --scale 1.05 --shift 0.08
```

Generate the quantile candidate:

```bash
python3 -m src.quantile_mae_candidate --scale 1.015 --shift 0.018
```

Submit to Kaggle:

```bash
kaggle competitions submit \
  -c cse-281-spring-26-item-price-prediction \
  -f submissions/cand_lasso_scale1p05_shift0p08/submission.csv \
  -m "Final calibrated lasso candidate"
```

Check submissions:

```bash
kaggle competitions submissions cse-281-spring-26-item-price-prediction
```

## What To Say In A Presentation

Use this 60-second story:

> The project is a tabular item price prediction problem where the target `Y`
> is already log-transformed and Kaggle evaluates MAE directly on `Y`. The main
> signal is item identity `X1`; almost all test items appear in training, so
> the task is mainly regularized item-mean estimation plus outlet and price
> adjustments. I built an sklearn pipeline with fold-safe feature engineering,
> target encoding for high-cardinality item IDs, and a compact <=5 model lineup.
> The best raw model was a lasso-style ElasticNet using smoothed target
> encoding. After confirming the metric was MAE, I added quantile regression as
> a metric-aligned model. It achieved the best local OOF MAE, but the public
> leaderboard favored calibrated lasso predictions. Small affine calibration of
> the lasso vector improved public LB from 0.378 to 0.370. I kept the final
> system small because extra models were highly correlated and did not add
> meaningful signal.

## Questions You May Be Asked

### Why not log-transform `Y`?

Because `Y` is already log-transformed in the provided dataset. Applying
another log transform would corrupt the target scale.

### What is the evaluation metric?

Kaggle reports Mean Absolute Error on the provided `Y` values.

### Why use target encoding?

`X1` is high-cardinality item identity. One-hot encoding every item is possible
but noisy and sparse; target encoding summarizes item-level expected target.
The implementation is cross-fitted to avoid target leakage.

### Is using test features in counts leakage?

No target labels from test are used. The pooled counts are unsupervised
statistics from feature columns only. This is transductive feature engineering,
not target leakage.

### Why does CV look worse than the public leaderboard?

Random CV folds contain many validation rows whose item-level evidence is weak
inside that fold. The Kaggle test/public split has very high item overlap with
training, so it is easier than random CV. Public LB is therefore lower than
local CV.

### Why did quantile regression not win publicly if MAE is the metric?

It did improve local MAE. However, the public leaderboard is one specific test
slice, and the calibrated lasso vector matched that slice better. This shows
that metric alignment is useful but not always sufficient.

### Why not keep the neural network?

The NN was slower, worse locally, and not competitive with the linear models.
Because the rules allow at most five models, it was removed from the active
registry when `quantile_mae` was added.

### What is the final model?

For leaderboard score, use the calibrated `lasso_te200` submission:

```text
submissions/cand_lasso_scale1p05_shift0p08/submission.csv
```

For a clean model explanation, describe:

- base model: `lasso_te200`
- post-processing: affine calibration with `scale=1.05`, `shift=0.08`
- public LB: `0.370`

### What is the biggest caveat?

Public leaderboard calibration can overfit the public split. It improved the
visible score, but a hidden/private split could prefer a less calibrated model
or the raw/near-raw lasso. For academic discussion, be transparent that the
calibrated submission is optimized for public LB, while local validation still
supports the simpler metric-aligned modeling story.

## Files To Open During A Code Walkthrough

Open these in this order:

1. `src/config.py`: target scale, paths, seed.
2. `src/features.py`: domain feature engineering.
3. `src/preprocessing.py`: leakage-safe preprocessing and target encoding.
4. `src/models.py`: final five model factories.
5. `src/train.py`: CV training loop and OOF artifacts.
6. `src/lb_calibration.py`: public-LB calibration logic.
7. `results/model_comparison.csv`: compact local model comparison.
8. `submissions/cand_lasso_scale1p05_shift0p08/note.txt`: final candidate note.

## Final Takeaways

- The winning approach is not a huge model zoo.
- The key is understanding the data split and item overlap.
- Target encoding is the central modeling technique.
- Lasso/ElasticNet is strong because the engineered features already expose the
  dominant signal.
- `quantile_mae` was the only new active model added; it replaced the NN to
  keep the <=5 cap.
- The best public LB result comes from calibrated lasso, not from adding more
  model complexity.

## What You Should Memorize Before The Uni Discussion

Memorize these points first:

1. The target `Y` is already log-transformed.
2. Kaggle evaluates MAE on `Y`.
3. The strongest feature is item identity `X1`.
4. Target encoding is used for high-cardinality item information.
5. Target encoding is done inside sklearn pipelines to avoid leakage.
6. The active final lineup has exactly five models.
7. Only one new active model was added late: `quantile_mae`.
8. `quantile_mae` replaced `nn` in the active registry.
9. `quantile_mae` had the best local OOF MAE.
10. Calibrated lasso had the best public leaderboard score.
11. Best public score reached: `0.370`.
12. Best submission file: `submissions/cand_lasso_scale1p05_shift0p08/submission.csv`.

If you get nervous, use this compact answer:

> We built a leakage-safe tabular ML pipeline. The most important signal is
> item ID, so we used target encoding plus regularized linear models. We added
> quantile regression because the metric is MAE, but the public leaderboard was
> best with calibrated lasso. The final active model set stays within the
> five-model cap.

## One-Minute Beginner Explanation

Here is the project explained as simply as possible:

> We have a spreadsheet of products sold in stores. For some rows we know the
> target value `Y`; for test rows we do not. The model learns from the known
> rows and predicts the missing values. The most useful clue is the item ID,
> because the same item often appears many times. We convert item IDs and other
> categories into useful numbers without leaking the answer, train several
> models, and compare them using cross-validation. A lasso model gave the best
> leaderboard prediction after a small adjustment. We also tried quantile
> regression because the metric is MAE; it was good locally, but not the best
> public submission.

## Two-Minute Technical Explanation

Here is the slightly more technical version:

> The repo implements an sklearn pipeline for a BigMart-style tabular
> regression task. `BigMartFeatures` performs cleaning, imputation, outlet/item
> feature engineering, frequency features, and interaction features. The
> preprocessing layer separates numeric, low-cardinality categorical, and
> high-cardinality target-encoded columns. Target encoding is cross-fitted and
> placed inside the pipeline, so validation folds do not see their own target
> values. The final model registry contains five models: lasso, quantile
> regression, random forest, LightGBM, and XGBoost. Quantile regression was
> added because MAE is the official metric, but the best public leaderboard
> result came from an affine-calibrated lasso prediction vector. The calibration
> improved public LB from 0.378 to 0.370, while the local comparison still shows
> quantile regression as the best OOF MAE model.
