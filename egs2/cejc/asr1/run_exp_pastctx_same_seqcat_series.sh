#!/usr/bin/env bash
# 同一話者・seqcat 方式（過去音声を連結）の N=1..5 系列。
#   pool_range=whole   NS="1 2 3 4 5" ./run_exp_pastctx_same_seqcat_series.sh   # 過去+現発話全体を pool（既定・まずこちら）
#   pool_range=current NS="1 2 3 4 5" ./run_exp_pastctx_same_seqcat_series.sh   # 現発話フレームのみ pool
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export max_past=${max_past:-30}
export pool_range=${pool_range:-whole}
for n in ${NS:-1 2 3 4 5}; do
    echo "===== same(seqcat/${pool_range}) N=${n} (max_past=${max_past}s) ====="
    scope=same N="${n}" max_past="${max_past}" ./run_seqcat.sh "$@"
done
