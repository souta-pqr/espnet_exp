#!/usr/bin/env bash
# 過去文脈：両話者（同一セッションの直前N発話、相手の発話も含む）。既定 N=2。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
context_scope=session N=${N:-2} exec ./run_ctxvec.sh "$@"
