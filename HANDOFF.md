# Session Handoff — CSE281 BigMart Item Price Prediction

**Last updated:** 2026-05-01 evening, after Push 1–8 work
**Repo:** https://github.com/Holmesberg/Item-Price-Prediction (private)
**Latest commit:** `62a8f21` (Push 8: scope-correction + X1 overlap diagnostic)

---

## TL;DR for the next session

- Best public LB: **0.379** (rank 9/19 of 17:55 UTC 2026-05-01).
- The X1=99.3% test/train overlap diagnostic is the most important thing in the project. Read `CLAUDE.md` § Diagnostics first.
- Stop adding models. 13 models already trained, OOF correlations are 0.97–1.00, stacking buys 0.001 OOF. Any further "model zoo" expansion is theater.
- Two submissions queued for next Kaggle window (`./auto_submit.sh` polls and fires automatically): Push-2 Ridge re-submit + 70/30 Ridge+LGBM hedge.
- **Open question to user:** what's the actual grade rubric? CLAUDE.md doesn't state weights for LB vs report. Email professor or check Moodle.
- Default plan if rubric unknown: Days 2–4 report fundamentals → Days 5–7 light tuning + KFold target encoding (only real upside lever left) → Day 8 exam prep → Days 9–10 buffer.

---

## Where we are: Kaggle leaderboard

```
Public LB top: 0.371 (4 teams tied for 1st-4th)
              0.372 (5th-6th)
              0.374 (7th)
              0.376 (8th)
              0.379 (us, 9th)         ← Push-2 Ridge alone
              0.380 (10th-13th)
              0.381 (14th-15th)
              0.383 (16th)            ← our Push-1 Ridge+XGB blend
              0.384 (17th)
              0.387 (18th)
              0.391 (19th)
```

**Submission limit reset:** 5/day, hit at 15:32 UTC 2026-05-01.
Reset window unknown — either at 00:00 UTC tonight or rolling 24h
from first sub (14:53 UTC May 2). `auto_submit.sh` polls every 60s.

---

## Diagnostics (the methodological contribution — see CLAUDE.md § Diagnostics)

1. **X1 overlap test↔train: 99.3%.** Only 17/2523 unseen items. Problem is per-item mean estimation, not feature engineering or model architecture. Any regularizer solves it.

2. **OOF correlations across 13 models: 0.97–1.00.** Every reasonable model converges to nearly the same prediction. Stacking is mathematically a weighted average of identical opinions.

3. **CV-LB gap: 0.5216 - 0.379 = 0.143, reverse direction (CV pessimistic).** CV folds over-sample rare-X1 rows; test set under-samples them. Not leakage. Test is genuinely easier than CV folds.

4. **Top-3 leaderboard tied at 0.371 = noise floor.** Compressed leaderboard reflects the irreducible variance in per-item mean estimation. Realistic ceiling for any pipeline ~0.371-0.375.

**Practical-exam answer template** for "why does your LB beat your CV":
> Test set has only 17/2523 unseen X1 values (0.7%); a random 5-fold CV split puts ~15–20% of validation rows on items seen <3 times in training. The model's hardest cases are over-represented in CV and under-represented in LB. Same reason every model in our lineup converged to nearly identical predictions — the problem is dominated by per-item mean estimation, which any reasonable regularizer solves. The leaderboard top-3 tied at 0.371 reflects the same noise floor.

---

## Pipeline / files

### `src/`

| File | Purpose | Status |
|---|---|---|
| `config.py` | SEED=42, paths, Y_LOG_MIN/MAX | stable |
| `features.py` | `BigMartFeatures` transformer with 19-column engineered output (5 raw num + X4_ratio + 4 counts + logX6 + X4_dev_X1 + 8 categoricals + X1 + X5_X11) | stable |
| `preprocessing.py` | `build_linear_preprocessor` (OHE+Scale+TargetEncoder), `build_tree_preprocessor` (Ordinal+TargetEncoder) | stable |
| `models.py` | 14 model factories — see lineup table below | stable |
| `nn_model.py` | PyTorch `TabularNN` + sklearn-compatible `SklearnNNRegressor` wrapper | stable |
| `nn_replicate.py` | Reproduces teacher's lab spec (val RMSE 0.5839 vs lab's 0.5834) | stable |
| `train.py` | `python -m src.train [model1 model2 ...]` — CLI for full or partial train | stable |
| `submit.py` | OOF-correlation-driven blend submission (sub_01_top + sub_02_blend) | stable |
| `optuna_lgbm.py` | Optuna TPE sweep for LightGBM, overwrites best_lgbm.joblib | done |
| `optuna_xgb.py` | Same for XGBoost | done |
| `catboost_model.py` | Standalone CatBoost with shared KFold | done |
| `decomp_experiment.py` | Y - log(X6) target experiment — falsified, didn't help | done |
| `pseudo_label.py` | Single-round pseudo-labeling (top 50% confident) | done |
| `pseudo_iter.py` | 3-round iterative pseudo-labeling | done |
| `stacking.py` | RidgeCV meta-learner | done |
| `stacking_v2.py` | NNLS-constrained + LGBM meta + mean-of-top-3 variants | done |
| `all_candidates.py` | Generates every submission variant from results/oof_*.npy | done |
| `eda.py` | 8 EDA figures (target dist, missingness, X3 dirty, X4 zeros, X6 vs Y, X1 prefix vs Y, outlet rollups, correlation) | done |
| `plots.py` | `feature_importance_lgbm.png` + `residuals_top_model.png` | done |
| `render_report.py` | Markdown → PDF via pandoc/xelatex (or weasyprint fallback) | done |

### `results/`

- `model_comparison.csv` — 14 rows, sorted by `cv_rmse_mean`
- `best_*.joblib` — 14 refit-on-full-train pipelines (one per model)
- `oof_*.npy` — 14 out-of-fold prediction arrays (each (6000,) float64)
- `optuna_lgbm_study.joblib`, `optuna_xgb_study.joblib` — Optuna study objects with full trial histories
- `teacher_replication.txt` — single line: 0.583900 (matches lab's 0.5834 within 0.001)
- `pseudo_ridge_test_pred.npy`, `pseudo_iter_test_pred.npy` — pseudo-labeled test predictions

### `submissions/`

- `sub_01_ridge/` — Push-2 Ridge submission, LB **0.379** (our best confirmed)
- `sub_02_blend_ridge_xgb/` — Push-2 Ridge+XGB blend (built but never uploaded — limit hit)
- `sub_02_blend_ridge_lgbm/` — Push-1 blend, LB 0.388 (Day 1 era)
- `cand_*/` — 11 candidates from the model-zoo phase (most are 0.97+ correlated → not real hedges)
- `cand_hedge_70r_30lgbm/` — the **only candidate worth submitting alongside Ridge alone** (70% Push-2 Ridge + 30% Optuna-LGBM, real if mathematically thin hedge)

Each submission directory has `submission.csv` + frozen `code/` snapshot + `requirements.txt`.

### `report/`

- `report.md` — section skeleton with placeholders for {{MODEL_COMPARISON_TABLE}}
- `report.pdf` — last rendered with Push-2 era data
- `figures/` — 10 figures (8 EDA + 2 diagnostics)

### Root

- `CLAUDE.md` — project context for Claude (now includes Diagnostics section)
- `requirements.txt` — `numpy>=2.0, pandas>=2.2, scikit-learn>=1.5, lightgbm>=4.5, matplotlib>=3.9, weasyprint>=63, torch>=2.5, optuna>=3.6` (also installed: xgboost, catboost, kaggle)
- `auto_submit.sh` — polls Kaggle every 60s, fires 2 submissions when window opens
- `train.csv`, `test.csv`, `sample_submission.csv` — competition data

---

## Model lineup (14 models, OOF RMSE log-Y)

| Model | OOF | Corr w/ Ridge | Notes |
|---|---|---|---|
| **enet** | **0.5207** | 0.9991 | Best single. ElasticNet alpha=0.003 l1_ratio=0.85 |
| ridge | 0.5216 | 1.0000 | Push-2 Ridge alone — known LB 0.379 |
| bridge | 0.5216 | 1.0000 | BayesianRidge, auto-tuned alpha |
| huber | 0.5238 | 0.9996 | HuberRegressor, outlier-robust loss |
| ridge_poly | 0.5232 | 0.9984 | Ridge with degree-2 interaction-only poly features |
| xgb | 0.5244 | 0.9965 | Optuna-tuned (50 trials), 0.5378 grid → 0.5224 → 0.5244 with new features |
| lgbm | 0.5271 | 0.9952 | Optuna-tuned (50 trials), 0.5347 grid → 0.5265 → 0.5270 with new features |
| lgbm_mae | 0.5323 | 0.9939 | LGBM with MAE objective — different loss, similar prediction |
| rf | 0.5335 | 0.9903 | RandomForest, max_depth=12, 500 trees |
| hgbr | 0.5341 | 0.9905 | sklearn HistGradientBoostingRegressor |
| knn | 0.5409 | 0.9869 | KNN, n_neighbors=50, distance-weighted |
| catboost | 0.5446 | 0.9833 | CatBoost defaults, 2000 iter |
| nn | 0.5482 | 0.9808 | PyTorch NN — most decorrelated peer |

(`bag_enet`/`bag_ridge` factories exist in models.py but their joblib/npy artifacts were removed because they hurt stacking.)

---

## Stacking results (do not pursue further)

| Stacking variant | OOF RMSE | Improvement over single Ridge (0.5216) |
|---|---|---|
| cand_stack_ridge (RidgeCV meta) | 0.52025 | -0.0014 (≈ -0.0009 expected on LB) |
| cand_stack_nnls (constrained, w≥0, sum=1) | 0.52061 | -0.0010 |
| cand_blend_top2 (inverse-RMSE of top 2) | 0.52083 | -0.0008 |
| cand_blend_uncorr | 0.52117 | -0.0004 |
| cand_mean_top3 | 0.52104 | -0.0006 |
| cand_mean (all 12) | 0.52294 | +0.0008 (worse) |
| cand_median | 0.52206 | -0.0001 |

Stacking does work, but the gain (~0.001 OOF) is dwarfed by the CV-LB gap and the noise floor. **Public LB will probably show ~0.378 for stack vs 0.379 for single Ridge.** Not worth the private-LB shake-up risk.

---

## Submission strategy (the only thing that matters from here)

**Floor:** Push-2 Ridge alone (`sub_01_ridge/submission.csv`) — known LB **0.379**.

**Insurance:** `cand_hedge_70r_30lgbm/submission.csv` — 70% Ridge + 30% Optuna-LGBM. Mathematically thin (0.995 corr × 30% weight ≈ ±0.0001 RMSE delta in expectation), but offers cheap insurance against tiny public→private shake-up.

**Auto-submit:** `./auto_submit.sh` (polls Kaggle daily limit every 60s, fires both candidates when window opens). Kaggle credential at `~/.kaggle/access_token` (KGAT_… token, valid for ~next 7 days per Ali's note).

**Final 2 picks for course evaluation:** by LB score, after both submissions confirmed. Not by CV. The CV-LB ordering on our 5 already-submitted singles is consistent (Push-2 wins both rankings), so CV is ordinally informative — but the absolute CV→LB gap is large and unpredictable.

**Do NOT submit:**
- `cand_stack_*` — overfits public LB at 0.99+ correlations; risks private shake-up
- `cand_pseudo_*` / `cand_calibrated_*` — pseudo-labeling on 99% overlap dataset just amplifies model bias on the rare 0.7%; calibration is variance shift on unknown distribution
- `cand_stack_lgbm` (LGBM meta) — self-fit OOF 0.49 is severely overfit; real test perf likely much worse

---

## What WORKED in this project (for the report)

1. **Day-1 baseline pipeline:** 5 models (Ridge, RF, LGBM, KNN, NN), 5-fold KFold CV, OOF-correlation-driven blend → LB 0.388.
2. **Push-2 features:** logX6, X1_count (train+test pooled), X7_count, X1_X7_count, X4_dev_X1 — moved Ridge from CV 0.5314 → 0.5216, LB 0.394 → **0.379**.
3. **sklearn TargetEncoder for X1** with internal cross-fitting — leakage-safe per-fold encoding of the 1553-unique-value categorical. Confirmed leakage-free by smoke test (val rows never see own Y).
4. **Optuna TPE** for LGBM and XGB — 50 trials each, found small-tree/low-lr regions for both.
5. **NN replication** of teacher's lab notebook — 0.5839 vs teacher's 0.5834 on identical 80/20 split. Sanity check that pipeline is faithful.

## What DID NOT WORK (for the report — these are negative results worth reporting)

1. **Ridge with degree-2 polynomial features** (ridge_poly): OOF 0.5240 vs Ridge 0.5216. Adding 36 pair interactions on top of 9 numerics increased noise faster than signal.
2. **Y-decomposition** (predict Y - log(X6)): OOF essentially identical to Ridge. Hypothesis was that decomposing the multiplicative MRP structure would help; falsified — the linear model already captured it via the logX6 feature.
3. **Bagged ENet / bagged Ridge** (15 bootstraps × 80% sample): OOF 0.5208 / 0.5214 vs non-bagged 0.5206 / 0.5215. Bagging slightly hurt — variance reduction didn't compensate for diluting Lasso-selected coefficients.
4. **CatBoost** (defaults, 2000 iter): OOF 0.5446 — ranks 12th of 14 models, worse than Ridge despite being a state-of-the-art GBM. Symmetric trees aren't a great fit for this problem.
5. **HuberRegressor**: OOF 0.5238 — outlier-robust loss didn't help on a clean log-transformed target.
6. **LightGBM with MAE objective**: OOF 0.5323. MAE optimizes median, RMSE evaluates mean → mismatch costs 0.005 OOF.
7. **Adding 8 models past Push 2:** total OOF gain across full stack from adding ridge_poly, enet, bridge, huber, hgbr, xgb, catboost, lgbm_mae was **0.0011** OOF. Hours of compute, near-zero LB return, because OOF correlations across all 13 models stayed at 0.97-1.00.

These negative results are cite-worthy in the report. Most teams won't report them.

---

## Open questions / action items for next session

1. **Rubric weights** (most important). CLAUDE.md doesn't state how LB rank vs report vs practical exam are weighted. Either email professor or check Moodle. Determines:
   - If LB-heavy: keep tuning + try Days 5-7 plan items (X6×X11, X3×X5 interactions)
   - If report-heavy: stop coding, write report around the 3 diagnostics (X1 overlap, OOF saturation, reverse CV-LB gap)
   - If even: submit the 2 queued, write report, light tuning between.

2. **Run `./auto_submit.sh`** when ready (or just on the Windows side — it polls patiently). Get LB scores for Push-2 Ridge (re-submit) and 70/30 hedge.

3. **Begin report draft.** Skeleton in `report/report.md` already exists from Day 1. Update §3 (Preprocessing) with the new features (logX6, counts, X4_dev_X1), update §4 (Models) — or trim to the rubric-required 4 — and most importantly, add a §3.5 or §11 that frames the X1 overlap diagnostic as the methodology contribution.

4. **Lock the top-2 submission selection on Kaggle** within 24h before the 2026-05-08 20:55 UTC deadline.

---

## Files NOT to touch (explicitly out of scope per established diagnostics)

- Don't add more models. Read `~/.claude/projects/-mnt-d-Studies-CSE281-Intro-to-AI-Project/memory/feedback_stop_when_oof_correlations_are_high.md` first.
- Don't add Co-Authored-By: Claude trailers to commits. Read `~/.claude/projects/-mnt-d-Studies-CSE281-Intro-to-AI-Project/memory/feedback_no_coauthor.md`.
- Don't paste secrets in chat. Use `! mkdir ... && echo TOKEN > ...` shell pattern. Read `~/.claude/projects/-mnt-d-Studies-CSE281-Intro-to-AI-Project/memory/feedback_secret_handling.md`.
- Don't fit target encoders globally before splitting. Always per-fold (sklearn TargetEncoder does this internally; verify if writing custom).

---

## Compute environment

- WSL2 Ubuntu, Python 3.12, no GPU (CPU torch).
- All packages in `~/.local/lib/python3.12/site-packages/` (installed with `pip3 --user --break-system-packages`).
- Pandas 3.0 reads strings as `str` dtype (not `object`); use `pd.api.types.is_numeric_dtype` not `dtype == "object"`.
- LightGBM requires `libgomp1` (apt-installed; user has sudo).
- Kaggle CLI at `~/.local/bin/kaggle`; auth at `~/.kaggle/access_token`.
- Pandoc + texlive-xetex installed for report PDF rendering.

---

## Memory files (auto-loaded into every session)

Located at `/home/alina/.claude/projects/-mnt-d-Studies-CSE281-Intro-to-AI-Project/memory/`:

- `MEMORY.md` — index
- `user_profile.md` — Ali, undergrad CSE281, AI/BCI trajectory, GPA-pressure
- `project_cse281.md` — project facts, deadline 2026-05-08, metric confirmed RMSE log-Y, Kaggle token location
- `project_x1_overlap_finding.md` — **the most important memory** — full X1=99.3% diagnosis
- `feedback_target_encoding.md` — TE per-fold rule
- `feedback_secret_handling.md` — `! shell` pattern for credentials
- `feedback_no_coauthor.md` — no Co-Authored-By Claude trailers
- `feedback_stop_when_oof_correlations_are_high.md` — stop adding models when OOF corr ≥ 0.95
