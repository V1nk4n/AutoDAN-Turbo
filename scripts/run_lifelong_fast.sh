#!/usr/bin/env bash
# Lifelong chunk only (không warm-up). Cần truyền chunk: 1, 2, 3, …
#
# Usage:
#   bash scripts/run_lifelong_fast.sh 1
#   LIFELONG_CHUNK=2 bash scripts/run_lifelong_fast.sh
set -euo pipefail
exec "$(dirname "$0")/run_pro_lifelong.sh" "$@"
