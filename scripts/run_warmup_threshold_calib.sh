#!/usr/bin/env bash
# Warm-up run for PRO threshold calibration (see docs/PRO_THRESHOLD_CALIBRATION.md).
set -euo pipefail
cd "$(dirname "$0")/.."

PATTERN_LIB="${PATTERN_LIB:-./logs/pattern_library_warmup_calib.json}"
EPOCHS="${EPOCHS:-50}"

python main.py \
  --pro_enabled \
  --only_warm_up \
  --epochs "$EPOCHS" \
  --log_every 5 \
  --use_local_embedding \
  --pattern_filepath "$PATTERN_LIB" \
  --pro_four_tier_eval \
  --pro_dynamic_pattern_select \
  --pro_per_candidate_strategy_bundles \
  --pro_explore_n_candidates 2 \
  --pro_exploit_n_candidates 4 \
  --pro_explore_max_new_tokens 512 \
  --pro_exploit_max_new_tokens 1024 \
  --pro_eval_batch_size 6 \
  --pro_tier1_min_response_chars 24 \
  --pro_goal_similarity_floor 0.15 \
  "$@"

echo ""
echo "Analyze: python scripts/analyze_pro_thresholds.py --latest --out threshold_report.md"
