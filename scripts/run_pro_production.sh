#!/usr/bin/env bash
# =============================================================================
# PRO production runner — chỉnh TOÀN BỘ tham số ở khối CONFIG bên dưới.
#
# Usage (staged — khuyến nghị production):
#   bash scripts/run_pro_warmup.sh              # warm-up → dừng
#   bash scripts/run_pro_lifelong.sh 1          # lifelong chunk 1 → dừng (không warm-up)
#   bash scripts/run_pro_lifelong.sh 2          # chunk 2 → dừng …
#
# Hoặc qua biến MODE:
#   bash scripts/run_pro_production.sh                    # MODE=warm_up (mặc định)
#   MODE=lifelong LIFELONG_CHUNK=1 bash scripts/run_pro_production.sh
#   MODE=smoke bash scripts/run_pro_production.sh
#
# Sau run:
#   python scripts/pro_smoke_gate.py --latest
#   python scripts/analyze_pro_thresholds.py --latest --out threshold_report_prod.md
#
# Lưu ý: PRO_USE_FAST_PROFILE=1 sẽ GHI ĐÈ một số flag CLI sau parse (xem pro_fast_profile.py).
#        Để mọi biến CONFIG có hiệu lực, để PRO_USE_FAST_PROFILE=0 (mặc định).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# RUN MODE
# -----------------------------------------------------------------------------
# warm_up | lifelong | smoke
#   warm_up  → --only_warm_up (thoát sau warm-up)
#   lifelong → --only_lifelong CHUNK (bỏ qua warm-up; chunk 1 cần warm_up_*.pkl)
MODE="${MODE:-warm_up}"
LIFELONG_CHUNK="${LIFELONG_CHUNK:-1}"
LIFELONG_TOTAL_CHUNKS="${LIFELONG_TOTAL_CHUNKS:-4}"  # ghi chú; khớp main.py --lifelong_iterations

# -----------------------------------------------------------------------------
# DATA / IO (generic defaults applied AFTER mode presets below)
# -----------------------------------------------------------------------------
DATA="${DATA:-./data/harmful_behavior_requests.json}"
USE_LOCAL_EMBEDDING="${USE_LOCAL_EMBEDDING:-1}"  # 1 = thêm --use_local_embedding

# -----------------------------------------------------------------------------
# THROUGHPUT PRESET
# -----------------------------------------------------------------------------
# 0 = dùng đúng các biến PRO_* bên dưới (khuyến nghị khi tune calib)
# 1 = bật --pro_fast_profile (ghi đè explore tokens, early_stop, dynamic pattern, …)
PRO_USE_FAST_PROFILE="${PRO_USE_FAST_PROFILE:-0}"

# -----------------------------------------------------------------------------
# NLL → score_loss (thang score_loss luôn 0–10)
# -----------------------------------------------------------------------------
NLL_MIN="${NLL_MIN:-2}"
NLL_MAX="${NLL_MAX:-5}"

# -----------------------------------------------------------------------------
# FOUR-TIER EVAL
# -----------------------------------------------------------------------------
PRO_FOUR_TIER_EVAL="${PRO_FOUR_TIER_EVAL:-1}"
PRO_VERIFIER_TOP_N="${PRO_VERIFIER_TOP_N:-1}"
PRO_ENABLE_EVAL_CACHE="${PRO_ENABLE_EVAL_CACHE:-1}"
PRO_EVAL_BATCH_SIZE="${PRO_EVAL_BATCH_SIZE:-2}"

# -----------------------------------------------------------------------------
# PHASE / CANDIDATES (mỗi repeat: explore HOẶC exploit)
# -----------------------------------------------------------------------------
PRO_PHASE_SPLIT="${PRO_PHASE_SPLIT:-0.7}"
PRO_EXPLORE_N_CANDIDATES="${PRO_EXPLORE_N_CANDIDATES:-2}"
PRO_EXPLORE_TOP_K="${PRO_EXPLORE_TOP_K:-1}"
PRO_EXPLOIT_N_CANDIDATES="${PRO_EXPLOIT_N_CANDIDATES:-4}"
PRO_EXPLOIT_TOP_K="${PRO_EXPLOIT_TOP_K:-2}"
PRO_EXPLORE_MAX_NEW_TOKENS="${PRO_EXPLORE_MAX_NEW_TOKENS:-64}"
PRO_EXPLOIT_MAX_NEW_TOKENS="${PRO_EXPLOIT_MAX_NEW_TOKENS:-128}"
PRO_N_CANDIDATES="${PRO_N_CANDIDATES:-4}"   # legacy baseline; phase dùng explore/exploit
PRO_TOP_K="${PRO_TOP_K:-2}"

# -----------------------------------------------------------------------------
# SEMANTIC PRUNE + GOAL GATE
# -----------------------------------------------------------------------------
PRO_GOAL_SIMILARITY_FLOOR="${PRO_GOAL_SIMILARITY_FLOOR:-0.15}"
PRO_GOAL_PRUNE_RELATIVE_RATIO="${PRO_GOAL_PRUNE_RELATIVE_RATIO:-0.90}"

# -----------------------------------------------------------------------------
# STRATEGY ATTRIBUTION (embedding)
# -----------------------------------------------------------------------------
PRO_STRATEGY_EMBED_MIN_SIM="${PRO_STRATEGY_EMBED_MIN_SIM:-0.28}"
PRO_STRATEGY_EMBED_MIN_MARGIN="${PRO_STRATEGY_EMBED_MIN_MARGIN:-0.05}"

# -----------------------------------------------------------------------------
# TIER1 / FAST JUDGE
# -----------------------------------------------------------------------------
PRO_TIER1_MIN_RESPONSE_CHARS="${PRO_TIER1_MIN_RESPONSE_CHARS:-24}"
PRO_FAST_JUDGE_MIN_LEN="${PRO_FAST_JUDGE_MIN_LEN:-24}"
# Fast judge & feedback scheduler: mặc định BẬT (không truyền --pro_disable_*)

# -----------------------------------------------------------------------------
# EARLY STOP (score_loss 0–10; CLI <1 được ×10 trong ProPipelineConfig)
# với NLL_MIN=2 NLL_MAX=5: 0.015 ≈ cùng độ nhạy NLL như 0.05 khi span NLL là [0,10]
# -----------------------------------------------------------------------------
PRO_EARLY_STOP_PATIENCE="${PRO_EARLY_STOP_PATIENCE:-8}"
PRO_EARLY_STOP_MIN_DELTA="${PRO_EARLY_STOP_MIN_DELTA:-0.015}"
PRO_FEEDBACK_EVERY="${PRO_FEEDBACK_EVERY:-2}"
PRO_FEEDBACK_COOLDOWN_REPEATS="${PRO_FEEDBACK_COOLDOWN_REPEATS:-1}"

# -----------------------------------------------------------------------------
# DYNAMIC PATTERN SELECT (chỉ có tác dụng khi PRO_DYNAMIC_PATTERN_SELECT=1)
# Calib run dùng 0.3 / 0.3 / 0.4; code default 0.25 / 0.25 / 0.50
# -----------------------------------------------------------------------------
PRO_DYNAMIC_PATTERN_SELECT="${PRO_DYNAMIC_PATTERN_SELECT:-0}"
PRO_PATTERN_EXPLOIT_N="${PRO_PATTERN_EXPLOIT_N:-3}"
PRO_PATTERN_EXPLORE_N="${PRO_PATTERN_EXPLORE_N:-2}"
PRO_PATTERN_RANK_W_RATE="${PRO_PATTERN_RANK_W_RATE:-0.3}"
PRO_PATTERN_RANK_W_AVG="${PRO_PATTERN_RANK_W_AVG:-0.3}"
PRO_PATTERN_RANK_W_REQ="${PRO_PATTERN_RANK_W_REQ:-0.4}"
PRO_PATTERN_RANK_LOW_RATE_PENALTY="${PRO_PATTERN_RANK_LOW_RATE_PENALTY:-0.85}"
PRO_PATTERN_RANK_LOW_RATE_MIN_TRIALS="${PRO_PATTERN_RANK_LOW_RATE_MIN_TRIALS:-3}"
PRO_PATTERN_EXPLORE_SEED="${PRO_PATTERN_EXPLORE_SEED:-}"  # để trống = không set seed

# -----------------------------------------------------------------------------
# OPTIONAL QUALITY (tắt mặc định — bật = chậm hơn)
# -----------------------------------------------------------------------------
PRO_PER_CANDIDATE_STRATEGY_BUNDLES="${PRO_PER_CANDIDATE_STRATEGY_BUNDLES:-0}"
PRO_ROTATE_EXPLORE_ACROSS_CANDIDATES="${PRO_ROTATE_EXPLORE_ACROSS_CANDIDATES:-0}"
PRO_STAGED_EVAL_ENABLED="${PRO_STAGED_EVAL_ENABLED:-0}"

# -----------------------------------------------------------------------------
# CACHE / TELEMETRY
# -----------------------------------------------------------------------------
PRO_RETRIEVAL_CACHE_MAX_ENTRIES="${PRO_RETRIEVAL_CACHE_MAX_ENTRIES:-4096}"
PRO_EVAL_CACHE_MAX_ENTRIES="${PRO_EVAL_CACHE_MAX_ENTRIES:-4096}"
TARGET_MAX_NEW_TOKENS="${TARGET_MAX_NEW_TOKENS:-150}"

# =============================================================================
# Mode presets — MUST run before EPOCHS/PATTERN_LIB/LOG_EVERY defaults (see below)
# =============================================================================
case "$MODE" in
  smoke)
    EPOCHS="${EPOCHS:-3}"
    PATTERN_LIB="${PATTERN_LIB:-./logs/pattern_library_smoke.json}"
    LOG_EVERY="${LOG_EVERY:-1}"
    ;;
  warm_up) ;;
  lifelong) ;;
  *)
    echo "Unknown MODE=$MODE (use warm_up|lifelong|smoke)" >&2
    exit 1
    ;;
esac

PATTERN_LIB="${PATTERN_LIB:-./logs/pattern_library.json}"
EPOCHS="${EPOCHS:-50}"
LOG_EVERY="${LOG_EVERY:-10}"

# =============================================================================
# Preflight (lifelong: đảm bảo có state trước khi chạy, không warm-up lại)
# =============================================================================
if [[ "$MODE" == "lifelong" ]]; then
  SUFFIX=""
  for _arg in "$@"; do
    if [[ "$_arg" == "--debug" ]]; then SUFFIX="_debug"; break; fi
  done
  if [[ "$LIFELONG_CHUNK" == "1" ]]; then
    NEED="./logs/warm_up_strategy_library${SUFFIX}.pkl"
    if [[ ! -f "$NEED" ]]; then
      echo "ERROR: warm-up state missing: $NEED" >&2
      echo "Run first: bash scripts/run_pro_warmup.sh" >&2
      exit 1
    fi
  else
    PREV=$((LIFELONG_CHUNK - 1))
    NEED="./logs/lifelong_strategy_library${SUFFIX}.pkl"
    if [[ ! -f "$NEED" ]]; then
      echo "ERROR: lifelong chunk $PREV results missing: $NEED" >&2
      echo "Run first: bash scripts/run_pro_lifelong.sh $PREV" >&2
      exit 1
    fi
  fi
fi

# =============================================================================
# Build CLI
# =============================================================================
EXTRA_MODE_ARGS=()
case "$MODE" in
  warm_up) EXTRA_MODE_ARGS+=(--only_warm_up) ;;
  lifelong) EXTRA_MODE_ARGS+=(--only_lifelong "$LIFELONG_CHUNK") ;;
  smoke) EXTRA_MODE_ARGS+=(--only_warm_up) ;;
esac

PRO_ARGS=(
  --pro_enabled
  --epochs "$EPOCHS"
  --log_every "$LOG_EVERY"
  --data "$DATA"
  --pattern_filepath "$PATTERN_LIB"
  --nll_min "$NLL_MIN"
  --nll_max "$NLL_MAX"
  --target_max_new_tokens "$TARGET_MAX_NEW_TOKENS"
  --pro_n_candidates "$PRO_N_CANDIDATES"
  --pro_top_k "$PRO_TOP_K"
  --pro_phase_split "$PRO_PHASE_SPLIT"
  --pro_explore_n_candidates "$PRO_EXPLORE_N_CANDIDATES"
  --pro_explore_top_k "$PRO_EXPLORE_TOP_K"
  --pro_exploit_n_candidates "$PRO_EXPLOIT_N_CANDIDATES"
  --pro_exploit_top_k "$PRO_EXPLOIT_TOP_K"
  --pro_explore_max_new_tokens "$PRO_EXPLORE_MAX_NEW_TOKENS"
  --pro_exploit_max_new_tokens "$PRO_EXPLOIT_MAX_NEW_TOKENS"
  --pro_eval_batch_size "$PRO_EVAL_BATCH_SIZE"
  --pro_goal_similarity_floor "$PRO_GOAL_SIMILARITY_FLOOR"
  --pro_goal_prune_relative_ratio "$PRO_GOAL_PRUNE_RELATIVE_RATIO"
  --pro_strategy_embed_min_sim "$PRO_STRATEGY_EMBED_MIN_SIM"
  --pro_strategy_embed_min_margin "$PRO_STRATEGY_EMBED_MIN_MARGIN"
  --pro_tier1_min_response_chars "$PRO_TIER1_MIN_RESPONSE_CHARS"
  --pro_fast_judge_min_len "$PRO_FAST_JUDGE_MIN_LEN"
  --pro_early_stop_patience "$PRO_EARLY_STOP_PATIENCE"
  --pro_early_stop_min_delta "$PRO_EARLY_STOP_MIN_DELTA"
  --pro_feedback_every "$PRO_FEEDBACK_EVERY"
  --pro_feedback_cooldown_repeats "$PRO_FEEDBACK_COOLDOWN_REPEATS"
  --pro_pattern_exploit_n "$PRO_PATTERN_EXPLOIT_N"
  --pro_pattern_explore_n "$PRO_PATTERN_EXPLORE_N"
  --pro_pattern_rank_w_rate "$PRO_PATTERN_RANK_W_RATE"
  --pro_pattern_rank_w_avg "$PRO_PATTERN_RANK_W_AVG"
  --pro_pattern_rank_w_req "$PRO_PATTERN_RANK_W_REQ"
  --pro_pattern_rank_low_rate_penalty "$PRO_PATTERN_RANK_LOW_RATE_PENALTY"
  --pro_pattern_rank_low_rate_min_trials "$PRO_PATTERN_RANK_LOW_RATE_MIN_TRIALS"
  --pro_retrieval_cache_max_entries "$PRO_RETRIEVAL_CACHE_MAX_ENTRIES"
  --pro_eval_cache_max_entries "$PRO_EVAL_CACHE_MAX_ENTRIES"
)

if [[ "$PRO_FOUR_TIER_EVAL" == "1" ]]; then
  PRO_ARGS+=(--pro_four_tier_eval)
fi
if [[ "$PRO_VERIFIER_TOP_N" != "2" ]]; then
  PRO_ARGS+=(--pro_verifier_top_n "$PRO_VERIFIER_TOP_N")
fi
if [[ "$PRO_ENABLE_EVAL_CACHE" == "1" ]]; then
  PRO_ARGS+=(--pro_enable_eval_cache)
fi
if [[ "$PRO_USE_FAST_PROFILE" == "1" ]]; then
  PRO_ARGS+=(--pro_fast_profile)
fi
if [[ "$PRO_DYNAMIC_PATTERN_SELECT" == "1" ]]; then
  PRO_ARGS+=(--pro_dynamic_pattern_select)
fi
if [[ "$PRO_PER_CANDIDATE_STRATEGY_BUNDLES" == "1" ]]; then
  PRO_ARGS+=(--pro_per_candidate_strategy_bundles)
fi
if [[ "$PRO_ROTATE_EXPLORE_ACROSS_CANDIDATES" == "1" ]]; then
  PRO_ARGS+=(--pro_rotate_explore_across_candidates)
fi
if [[ "$PRO_STAGED_EVAL_ENABLED" == "1" ]]; then
  PRO_ARGS+=(--pro_staged_eval_enabled)
fi
if [[ -n "$PRO_PATTERN_EXPLORE_SEED" ]]; then
  PRO_ARGS+=(--pro_pattern_explore_seed "$PRO_PATTERN_EXPLORE_SEED")
fi
if [[ "$USE_LOCAL_EMBEDDING" == "1" ]]; then
  PRO_ARGS+=(--use_local_embedding)
fi

echo "=== PRO run_production ==="
echo "  MODE=$MODE  EPOCHS=$EPOCHS  PATTERN_LIB=$PATTERN_LIB"
if [[ "$MODE" == "lifelong" ]]; then
  echo "  LIFELONG_CHUNK=$LIFELONG_CHUNK (only this chunk; warm-up skipped)  total_chunks≈$LIFELONG_TOTAL_CHUNKS"
fi
if [[ "$MODE" == "warm_up" ]]; then
  echo "  Stops after warm-up. Next: bash scripts/run_pro_lifelong.sh 1"
fi
echo "  NLL=[$NLL_MIN,$NLL_MAX]  four_tier=$PRO_FOUR_TIER_EVAL  fast_profile=$PRO_USE_FAST_PROFILE"
echo "  pattern_w=(${PRO_PATTERN_RANK_W_RATE},${PRO_PATTERN_RANK_W_AVG},${PRO_PATTERN_RANK_W_REQ})  dynamic_pattern=$PRO_DYNAMIC_PATTERN_SELECT"
echo "  early_stop: patience=$PRO_EARLY_STOP_PATIENCE  min_delta=$PRO_EARLY_STOP_MIN_DELTA (CLI; effective ×10 if <1)"
echo ""

python main.py \
  "${PRO_ARGS[@]}" \
  "${EXTRA_MODE_ARGS[@]}" \
  "$@"

echo ""
echo "Post-run:"
echo "  python scripts/pro_smoke_gate.py --latest"
echo "  python scripts/analyze_pro_thresholds.py --latest --out threshold_report_prod.md"
