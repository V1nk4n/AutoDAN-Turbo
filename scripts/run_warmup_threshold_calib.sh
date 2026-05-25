#!/usr/bin/env bash
# Warm-up calibration run (see docs/PRO_WARMUP_STEP_BY_STEP.md).
set -euo pipefail
cd "$(dirname "$0")/.."

PATTERN_LIB="${PATTERN_LIB:-./logs/pattern_library_calib_v4.json}"
EPOCHS="${EPOCHS:-15}"

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
  --pro_explore_n_candidates 4 \
  --pro_exploit_n_candidates 6 \
  --pro_explore_top_k 2 \
  --pro_exploit_top_k 3 \
  --pro_phase_split 0.30 \
  --pro_early_stop_patience 8 \
  --pro_early_stop_min_delta 0.05 \
  --pro_explore_max_new_tokens 512 \
  --pro_exploit_max_new_tokens 1024 \
  --pro_eval_batch_size 6 \
  --pro_tier1_min_response_chars 24 \
  --pro_goal_similarity_floor 0.15 \
  --pro_pattern_rank_w_rate 0.3 \
  --pro_pattern_rank_w_avg 0.3 \
  --pro_pattern_rank_w_req 0.4 \
  "$@"

echo ""
echo "Gate check:  python scripts/pro_smoke_gate.py --latest"
echo "Full report: python scripts/analyze_pro_thresholds.py --latest --out threshold_report_calib.md"
