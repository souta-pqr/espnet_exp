#!/usr/bin/env bash
# 過去文脈：過去の類似発話 top_k を集めて平均（kNN・同一セッションの過去発話のみ）。既定 top_k=5。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
context_scope=cluster top_k=${top_k:-5} exec ./run_ctxvec.sh "$@"
