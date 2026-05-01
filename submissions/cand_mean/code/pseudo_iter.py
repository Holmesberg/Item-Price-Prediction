"""Iterative pseudo-labeling: 3 rounds, expanding confidence threshold each
time. Round 1 uses top-40% confidence → trains Ridge+LGBM+XGB on augmented
data → predicts test with ensemble. Round 2 uses top-60% (more lenient
because the model is now stronger). Round 3 uses top-80%.

Each round saves a checkpoint test prediction. The final round's
predictions (or a blend across rounds) become a candidate.

Usage: python -m src.pseudo_iter
Output: results/pseudo_iter_test_pred.npy + submissions/cand_pseudo_iter/
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import (
    RESULTS_DIR,
    ROOT,
    SEED,
    SUBMISSIONS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    Y_LOG_MAX,
    Y_LOG_MIN,
    set_global_seed,
)
from .features import BigMartFeatures
from .models import build_enet, build_lgbm, build_ridge, build_xgb


def _ensemble_predict(test_df, pipes_with_weights):
    cols = [w * pipe.predict(test_df) for pipe, w in pipes_with_weights]
    return sum(cols)


def _ensemble_std(test_df, pipes):
    """Std across raw model predictions on test."""
    preds = np.column_stack([p.predict(test_df) for p in pipes])
    return preds.std(axis=1)


def main() -> None:
    set_global_seed(SEED)
    test_df = pd.read_csv(TEST_CSV)
    n_test = len(test_df)
    BigMartFeatures.EXTRA_COUNT_REF = test_df

    train = pd.read_csv(TRAIN_CSV)
    X_real, y_real = train.drop(columns=["Y"]), train["Y"].astype(float)

    # Build the round-1 ensemble: load existing best ridge, lgbm, xgb,
    # blend with their OOF-RMSE-inverse weights from results/.
    base = []
    weights = []
    for name in ["enet", "lgbm", "xgb"]:
        pipe = joblib.load(RESULTS_DIR / f"best_{name}.joblib")
        oof = np.load(RESULTS_DIR / f"oof_{name}.npy")
        rmse = float(np.sqrt(((oof - y_real.values) ** 2).mean()))
        base.append((pipe, name, rmse))
        weights.append(1.0 / rmse)
    weights = np.array(weights) / np.sum(weights)
    print("Round-0 ensemble (existing best models):")
    for (_, name, rmse), w in zip(base, weights):
        print(f"  {name:<6} OOF RMSE {rmse:.4f}  weight {w:.3f}")

    # Round-0 test predictions
    test_pred_iter = _ensemble_predict(
        test_df, [(p, w) for (p, _, _), w in zip(base, weights)]
    )
    raw_pipes = [p for p, _, _ in base]
    test_std = _ensemble_std(test_df, raw_pipes)

    rounds = [(0.4, "round1"), (0.6, "round2"), (0.8, "round3")]
    iter_history = []
    for q, label in rounds:
        threshold = float(np.quantile(test_std, q))
        confident = test_std <= threshold
        n_pseudo = int(confident.sum())
        print(f"\n=== {label}: top {q:.0%} confident ({n_pseudo}/{n_test}, std<={threshold:.4f}) ===")

        X_pseudo = test_df.loc[confident].copy()
        y_pseudo = pd.Series(test_pred_iter[confident], index=X_pseudo.index, name="Y")
        X_aug = pd.concat([X_real, X_pseudo], ignore_index=True)
        y_aug = pd.concat([y_real, y_pseudo], ignore_index=True)
        print(f"  augmented training rows: {len(X_aug)}")

        # Retrain ridge, lgbm, xgb on augmented data
        new_pipes = []
        new_rmses = []
        for name, factory in [("enet", build_enet), ("lgbm", build_lgbm), ("xgb", build_xgb)]:
            t0 = time.perf_counter()
            pipe = factory(seed=SEED)
            # Use the same hyperparams as our best from results/
            best_pipe = joblib.load(RESULTS_DIR / f"best_{name}.joblib")
            best_params = best_pipe.named_steps["model"].get_params()
            # Remove non-init params that can't be set on a fresh pipeline
            try:
                pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()
                                   if k in pipe.named_steps["model"].get_params()})
            except Exception:
                pass
            pipe.fit(X_aug, y_aug)
            elapsed = time.perf_counter() - t0
            test_p = pipe.predict(test_df)
            # Approximate "RMSE" using the original training data only
            train_pred = pipe.predict(X_real)
            train_rmse = float(np.sqrt(((train_pred - y_real.values) ** 2).mean()))
            print(f"  {name}: fit {elapsed:.1f}s  | train RMSE {train_rmse:.4f}")
            new_pipes.append(pipe)
            new_rmses.append(train_rmse)

        # New weights (still inverse-RMSE on train)
        w_new = np.array([1.0 / r for r in new_rmses])
        w_new /= w_new.sum()
        test_pred_iter = sum(w * pipe.predict(test_df) for pipe, w in zip(new_pipes, w_new))
        test_std = _ensemble_std(test_df, new_pipes)
        iter_history.append((label, test_pred_iter.copy()))

    # Save final iterated prediction
    final_pred = iter_history[-1][1]
    out_dir = SUBMISSIONS_DIR / "cand_pseudo_iter"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": final_pred}).to_csv(
        out_dir / "submission.csv", index=False
    )
    (out_dir / "note.txt").write_text(
        f"3-round iterative pseudo-labeling.\n"
        f"Final ensemble of ridge+lgbm+xgb retrained 3 times on growing pseudo-aug train.\n"
        f"Pred range: [{final_pred.min():.3f}, {final_pred.max():.3f}]\n",
        encoding="utf-8",
    )
    code = out_dir / "code"
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(
        ROOT / "src", code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(ROOT / "requirements.txt", out_dir / "requirements.txt")

    np.save(RESULTS_DIR / "pseudo_iter_test_pred.npy", final_pred)
    print(f"\nWrote {out_dir}/submission.csv")

    # Also write a 50/50 blend with the original cand_stack_ridge
    stack_path = SUBMISSIONS_DIR / "cand_stack_ridge" / "submission.csv"
    if stack_path.exists():
        stack_pred = pd.read_csv(stack_path)["Y"].values
        blend = 0.5 * stack_pred + 0.5 * final_pred
        out2 = SUBMISSIONS_DIR / "cand_stack_pseudo_iter"
        out2.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_id": np.arange(n_test, dtype=int), "Y": blend}).to_csv(
            out2 / "submission.csv", index=False
        )
        (out2 / "note.txt").write_text(
            "50/50 blend of cand_stack_ridge and cand_pseudo_iter.\n", encoding="utf-8"
        )
        code2 = out2 / "code"
        if code2.exists():
            shutil.rmtree(code2)
        shutil.copytree(
            ROOT / "src", code2, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        shutil.copy2(ROOT / "requirements.txt", out2 / "requirements.txt")
        print(f"Wrote {out2}/submission.csv")


if __name__ == "__main__":
    main()
