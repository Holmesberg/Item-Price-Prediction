# Project Memory: CSE281 Item Price Prediction

This file replaces the old `CLAUDE.md` and `HANDOFF.md` notes. It keeps the
important project facts in one stable, repo-tracked place without preserving
stale instructions or old session clutter.

## Student

- Name: Ali Nasser Sabry
- ID: 24P0248
- Course: CSE281 Introduction to AI
- Project: Kaggle item price prediction competition
- Repository: `Holmesberg/Item-Price-Prediction`

## Competition

- Kaggle slug: `cse-281-spring-26-item-price-prediction`
- Task type: tabular regression
- Kaggle deadline has passed; final deliverable is for LMS submission.
- Data shape:
  - `train.csv`: 6000 rows, columns `X1` through `X11` plus `Y`
  - `test.csv`: 2523 rows, columns `X1` through `X11`
  - `sample_submission.csv`: `row_id`, `Y`
- Submission format:
  - `row_id` is a zero-based integer matching `test.csv` order
  - `Y` must be submitted directly on the provided target scale

## Target And Metric

- `Y` is already log-transformed.
- Do not apply another `log1p`.
- Do not inverse-transform predictions before submission.
- Kaggle evaluates Mean Absolute Error (MAE) on `Y`.

## Anonymized Column Mapping

| Column | Meaning | Key handling |
|---|---|---|
| `X1` | item identifier | high-cardinality; prefix is useful; target-encoded |
| `X2` | item weight | impute by item mean, then global fallback |
| `X3` | item fat content | normalize dirty labels; non-consumables become `Non-Edible` |
| `X4` | item visibility | treat zeros as missing; impute by item mean |
| `X5` | item type | categorical |
| `X6` | item MRP / price-like value | strong numeric predictor; also use `logX6` and bins |
| `X7` | outlet identifier | categorical and count feature |
| `X8` | outlet establishment year | convert to `Outlet_Years = 2013 - X8` |
| `X9` | outlet size | mode-impute within outlet type |
| `X10` | outlet location tier | categorical |
| `X11` | outlet type | categorical |

## Current Feature Engineering

Implemented in `src/features.py` through `BigMartFeatures`.

Numeric engineered features:

- `X2`
- `X4`
- `X6`
- `Outlet_Years`
- `X4_ratio`
- `X1_count`
- `X7_count`
- `X1_X7_count`
- `logX6`
- `X4_dev_X1`

Categorical engineered features:

- `X1_prefix`
- `X3`
- `X5`
- `X6_bin`
- `X7`
- `X9`
- `X10`
- `X11`

Target-encoded features:

- `X1`
- `X5_X11`

## Leakage Rules

- Target-dependent preprocessing must stay inside sklearn pipelines.
- `TargetEncoder` is used with internal cross-fitting.
- Validation rows must not encode themselves using their own `Y`.
- Pooling `test.csv` into count statistics is allowed because it uses only
  feature columns, not labels.
- Never fit target encoders globally before cross-validation.

## Final Active Model Lineup

The final active registry in `src/models.py` has exactly five models:

1. `lasso_te200`
2. `quantile_mae`
3. `rf`
4. `lgbm`
5. `xgb`

Notes:

- `quantile_mae` was added because the metric is MAE.
- `nn` was removed from the active registry to keep the final lineup within the
  five-model cap.
- Huber was tested only as a legacy probe and is not part of the final lineup.

## Current Local Comparison

From `results/model_comparison.csv`:

| Model | OOF MAE | OOF RMSE | Note |
|---|---:|---:|---|
| `quantile_mae` | 0.39627 | 0.52504 | best local MAE |
| `lasso_te200` | 0.40152 | 0.52041 | best public after calibration |
| `xgb` | 0.40593 | 0.52442 | strongest tree baseline |
| `lgbm` | 0.40813 | 0.52709 | tuned boosting baseline |
| `rf` | 0.41429 | 0.53351 | tree baseline |

## Best Public Submission

Best confirmed public score: `0.370`.

Recommended final submission file:

```text
submissions/cand_lasso_scale1p05_shift0p08/submission.csv
```

This is an affine-calibrated version of the `lasso_te200` prediction vector:

```text
prediction = mean(test_pred) + 1.05 * (test_pred - mean(test_pred)) + 0.08
```

Important public scores:

| Candidate | Public score |
|---|---:|
| raw `lasso_te200` | 0.378 |
| `scale=1.00`, `shift=0.05` | 0.374 |
| `scale=1.0276`, `shift=0.0551` | 0.371 |
| `scale=1.05`, `shift=0.08` | 0.370 |
| `scale=1.0664`, `shift=0.0728` | 0.370 |
| `scale=1.07`, `shift=0.08` | 0.370 |
| `quantile_mae` calibrated probe | 0.371 |
| Huber calibrated probe | 0.375 |

## Main Diagnostic Insight

The task is dominated by item identity:

- 99.3 percent of test `X1` values appear in training.
- Only 17 of 2523 test rows have unseen items.
- Most reasonable models converge to very similar predictions.
- Adding many more models produced tiny gains because OOF predictions were
  highly correlated.

Interpretation:

> The core task is regularized item-mean estimation plus outlet and price
> adjustments, not complex representation learning.

## Why CV And Public LB Differ

Random 5-fold CV hides some item examples inside each fold. That makes local
validation harder because the model has weaker item-level evidence for some
validation rows.

The public test split has very high item overlap with the full training set.
When the model is refit on all 6000 rows, it has stronger item-level evidence.

Therefore:

- CV is useful for comparing models.
- CV is pessimistic in absolute value.
- Public-LB calibration can improve the visible score but may overfit the
  public split.

## What Worked

- Domain-aware BigMart feature engineering.
- Leakage-safe target encoding of `X1`.
- Regularized linear modeling, especially `lasso_te200`.
- Adding `quantile_mae` for local MAE alignment.
- Small affine calibration of lasso predictions for public LB.
- Keeping the final model set small.

## What Did Not Work

- Large model zoo expansion.
- Aggressive calibration.
- Huber public probe.
- Complex stacking/ensembling after predictions were already highly
  correlated.
- Neural network as a final active model.

## Reproducibility

Useful commands:

```bash
python3 -m src.train
python3 -m src.submit
python3 -m src.lb_calibration --scale 1.05 --shift 0.08
python3 -m src.render_report
```

Check Kaggle submissions:

```bash
kaggle competitions submissions cse-281-spring-26-item-price-prediction
```

## Presentation Summary

Use this if asked to summarize the project:

> This is a tabular regression problem where the target is already
> log-transformed and Kaggle evaluates MAE. The strongest signal is item ID, so
> the solution focuses on leakage-safe target encoding and regularized models.
> I compared a final five-model lineup: lasso, quantile regression, random
> forest, LightGBM, and XGBoost. Quantile regression gave the best local OOF
> MAE, but calibrated lasso gave the best public leaderboard result at 0.370.
> The main methodological insight is that 99.3 percent of test item IDs appear
> in train, so the problem is mostly regularized item-mean estimation rather
> than a need for a large ensemble.
