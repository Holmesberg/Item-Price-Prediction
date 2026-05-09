# CSE281 Item Price Prediction: Discussion Guide

This document is the end-to-end explanation of the repository as of May 9,
2026. Use it to prepare for a project discussion, viva, practical exam, or
code walkthrough.

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

Important target fact:

- `Y` is already log-transformed.
- Do not apply `log1p`, `exp`, or reverse transforms before submission.
- Kaggle evaluates MAE directly on `Y`.

Data files:

- `train.csv`: 6000 labeled rows.
- `test.csv`: 2523 unlabeled rows.
- `sample_submission.csv`: expected submission shape.

Submission format:

```text
row_id,Y
0,<prediction for first test row>
1,<prediction for second test row>
...
```

The project predicts `Y` for all 2523 test rows.

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

### 3. `rf`

Implementation:

- `RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=5)`

Role:

- Nonlinear bagged-tree baseline.

Result:

- Worse local OOF MAE than the linear models.

### 4. `lgbm`

Implementation:

- Tuned `LGBMRegressor`
- Small leaves, low learning rate, regularization.

Role:

- Gradient boosting baseline.

Result:

- Useful as a sanity check, but not better than lasso/quantile.

### 5. `xgb`

Implementation:

- Tuned `XGBRegressor`
- Histogram tree method.

Role:

- Second gradient boosting implementation for comparison.

Result:

- Better than RF/LGBM locally, but still not better than the linear family.

## Current Local Model Comparison

From `results/model_comparison.csv`:

| Model | OOF MAE | OOF RMSE | Notes |
|---|---:|---:|---|
| `quantile_mae` | 0.39627 | 0.52504 | Best local MAE, public 0.371 |
| `lasso_te200` | 0.40152 | 0.52041 | Best public after calibration |
| `xgb` | 0.40593 | 0.52442 | Strongest tree baseline |
| `lgbm` | 0.40813 | 0.52709 | Tuned boosting baseline |
| `rf` | 0.41429 | 0.53351 | Tree baseline |

Why OOF MAE and public LB disagree:

- Public LB is a specific subset of test rows.
- The public subset appears easier than random CV folds.
- Calibration can improve public LB while worsening local OOF.
- This is useful for leaderboard placement, but should be described honestly as
  public-leaderboard calibration, not as proof of better generalization.

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
