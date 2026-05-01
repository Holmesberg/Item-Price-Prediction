#!/bin/bash
# Auto-submit when Kaggle daily limit reopens.
#
# Conservative 2-submission plan (after the X1=99.3%-overlap diagnostic
# revealed the dataset is per-item-mean-estimation, not feature-engineering):
#
#   1. Push-2 Ridge alone (re-submit) — safe floor, known LB 0.379, simplest
#      reproducible model.
#   2. 70/30 hedge of Push-2 Ridge + Optuna-LGBM — ~the only meaningfully
#      decorrelated peer (OOF corr 0.995). Bias toward Ridge floor; let LGBM
#      catch rows where Ridge misses.
#
# Both are picked AS HEDGES, not by best-CV. We deliberately avoid:
#   - cand_stack_ridge / cand_stack_nnls (overfits public LB at 0.99+ corrs)
#   - cand_pseudo_iter / cand_stack_pseudo_iter (pseudo-labeling on a 99%
#     overlap dataset just amplifies model bias on the rare rows)
#   - cand_calibrated_stack (variance shift of unknown distribution)
#
# Final 2 picks for course evaluation will be made by LB score, not CV.

set -u
cd "$(dirname "$0")"
COMP="cse-281-spring-26-item-price-prediction"
KAGGLE="$HOME/.local/bin/kaggle"

echo "[auto_submit] Polling Kaggle daily limit (every 60s)..."
while true; do
  out=$($KAGGLE competitions submit -c "$COMP" -f submissions/sub_01_ridge/submission.csv \
        -m "Push-2 Ridge alone (re-submit, expect LB ~0.379)" 2>&1)
  if echo "$out" | grep -q "Successfully submitted"; then
    echo "[auto_submit] Window open — Ridge submitted."
    break
  fi
  echo "[auto_submit] Locked: $(echo "$out" | tail -1)"
  sleep 60
done

# Fire the hedge.
echo "[auto_submit] Submitting cand_hedge_70r_30lgbm..."
$KAGGLE competitions submit -c "$COMP" \
  -f submissions/cand_hedge_70r_30lgbm/submission.csv \
  -m "70/30 Ridge+LGBM hedge — OOF 0.5218, only meaningfully decorrelated peer" 2>&1 | tail -2

echo "[auto_submit] Done. Fetching scores in 30s..."
sleep 30
$KAGGLE competitions submissions -c "$COMP" 2>&1 | head -5

echo
echo "Final 2 picks for course evaluation should be chosen by LB score:"
echo "  1. Push-2 Ridge alone (LB confirmed 0.379 floor)"
echo "  2. Whichever of {cand_hedge_70r_30lgbm, Push-1 Ridge+XGB blend at 0.383}"
echo "     scores better LB. They're both Ridge-anchored hedges."
