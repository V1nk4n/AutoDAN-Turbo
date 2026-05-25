#!/usr/bin/env bash
# PRO lifelong chunk N — KHÔNG chạy lại warm-up (main.py: --only_lifelong N).
#
# Usage:
#   bash scripts/run_pro_lifelong.sh 1    # cần logs/warm_up_strategy_library.pkl
#   bash scripts/run_pro_lifelong.sh 2    # cần kết quả lifelong chunk 1
#   LIFELONG_CHUNK=3 bash scripts/run_pro_lifelong.sh
#
# Mặc định 4 chunk (khớp --lifelong_iterations trong main.py).
set -euo pipefail

CHUNK="${LIFELONG_CHUNK:-${1:-}}"
if [[ -z "$CHUNK" ]]; then
  echo "Usage: bash scripts/run_pro_lifelong.sh <chunk>" >&2
  echo "  chunk: 1-indexed lifelong iteration (1, 2, 3, …)" >&2
  echo "  Chunk 1 loads warm_up; chunk 2+ loads previous lifelong logs." >&2
  exit 1
fi

if ! [[ "$CHUNK" =~ ^[0-9]+$ ]] || [[ "$CHUNK" -lt 1 ]]; then
  echo "Invalid LIFELONG_CHUNK=$CHUNK (must be integer >= 1)" >&2
  exit 1
fi

exec env MODE=lifelong LIFELONG_CHUNK="$CHUNK" "$(dirname "$0")/run_pro_production.sh" "$@"
