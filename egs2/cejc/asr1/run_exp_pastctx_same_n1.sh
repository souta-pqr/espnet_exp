#!/usr/bin/env bash
# 過去文脈：同一話者の直前1発話（same, N=1）。新規 ctx_vec_same_n1 を抽出して学習。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
context_scope=same N=1 exec ./run_ctxvec.sh "$@"
