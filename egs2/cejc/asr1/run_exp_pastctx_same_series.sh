#!/usr/bin/env bash
# 同一話者の過去発話を増やした系列を順に学習する（feature-vector・ASR不変・未来なし）。
# 既定: N=3,4,5 を max_past=30秒 で。N を増やすには窓を広げないと実際に増えない（10秒だと平均2発話）。
#   NS="1 2 3 4 5" max_past=30 ./run_exp_pastctx_same_series.sh  # クリーンな N系列を一括
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export max_past=${max_past:-30}
for n in ${NS:-3 4 5}; do
    echo "===== same N=${n} (max_past=${max_past}s) ====="
    context_scope=same N="${n}" max_past="${max_past}" ./run_ctxvec.sh "$@"
done
