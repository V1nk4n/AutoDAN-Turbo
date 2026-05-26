#!/usr/bin/env bash
# Đánh giá PRO sau warm-up / lifelong — config khớp run_pro_production.sh
#
# Usage:
#   bash scripts/run_pro_eval.sh                    # lifelong, 10 repeat/request
#   bash scripts/run_pro_eval.sh warm_up            # split warm_up
#   EVAL_EPOCHS=5 MAX_REQUESTS=20 bash scripts/run_pro_eval.sh
#   bash scripts/run_pro_eval.sh --use_harmbench_classifier
set -euo pipefail
cd "$(dirname "$0")/.."

SPLIT="${1:-lifelong}"
shift 1 2>/dev/null || true

EVAL_EPOCHS="${EVAL_EPOCHS:-10}"
MAX_REQUESTS="${MAX_REQUESTS:-}"
PATTERN_LIB="${PATTERN_LIB:-./logs/pattern_library.json}"
SAVE_JSON="${SAVE_JSON:-./logs/eval_pro_${SPLIT}.json}"

EXTRA=()
if [[ -n "$MAX_REQUESTS" ]]; then
  EXTRA+=(--max_requests "$MAX_REQUESTS")
fi

python eval_pro.py \
  --production \
  --split "$SPLIT" \
  --pattern_filepath "$PATTERN_LIB" \
  --eval_epochs "$EVAL_EPOCHS" \
  --use_local_embedding \
  --save_json "$SAVE_JSON" \
  "${EXTRA[@]}" \
  "$@"
