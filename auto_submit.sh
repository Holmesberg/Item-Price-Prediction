#!/bin/bash
# Auto-submit the top 5 candidates the moment Kaggle daily limit reopens.
# Polls every 60s with cand_stack_ridge (cheap signal) until 200 OK, then
# fires the rest in sequence. Stop with Ctrl-C.

set -u
cd "$(dirname "$0")"
COMP="cse-281-spring-26-item-price-prediction"
KAGGLE="$HOME/.local/bin/kaggle"

CANDS=(
  "cand_stack_ridge:Push7: stack RidgeCV meta over 12 base (OOF 0.52025)"
  "cand_blend_top2:Push7: top-2 inverse-RMSE blend (OOF 0.52083)"
  "cand_stack_nnls:Push7: SLSQP-constrained meta (enet-dominant, OOF 0.52061)"
  "cand_pseudo_iter:Push7: 3-round iterative pseudo-labeling"
  "cand_calibrated_stack:Push7: variance-calibrated stack (mean=7.30 std=1.01)"
)

echo "[auto_submit] Polling Kaggle daily limit (every 60s)..."
while true; do
  out=$($KAGGLE competitions submit -c "$COMP" -f submissions/cand_stack_ridge/submission.csv \
        -m "Push7: stack RidgeCV meta over 12 base (OOF 0.52025)" 2>&1)
  if echo "$out" | grep -q "Successfully submitted"; then
    echo "[auto_submit] Window open!"
    break
  fi
  echo "[auto_submit] Still locked: $(echo "$out" | tail -1)"
  sleep 60
done

# First was already used as the probe. Fire the remaining 4.
for entry in "${CANDS[@]:1}"; do
  cand="${entry%%:*}"
  msg="${entry#*:}"
  echo "[auto_submit] Submitting $cand"
  $KAGGLE competitions submit -c "$COMP" -f "submissions/$cand/submission.csv" -m "$msg" 2>&1 | tail -2
  sleep 5
done

echo "[auto_submit] Done. Fetching scores..."
sleep 30
$KAGGLE competitions submissions -c "$COMP" 2>&1 | head -10
