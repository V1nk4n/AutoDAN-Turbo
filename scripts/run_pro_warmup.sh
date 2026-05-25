#!/usr/bin/env bash
# PRO warm-up only — chạy xong thì thoát, lưu state vào logs/warm_up_*.
#
# Usage:
#   bash scripts/run_pro_warmup.sh
#   bash scripts/run_pro_warmup.sh --debug
#
# Bước tiếp theo (sau khi warm-up xong):
#   bash scripts/run_pro_lifelong.sh 1
set -euo pipefail
exec env MODE=warm_up "$(dirname "$0")/run_pro_production.sh" "$@"
