#!/usr/bin/env bash
# 複数話者(session)の過去発話を増やした系列を順に学習する（feature-vector・ASR不変・未来なし）。
# session = reco から _IC* を除いたセッション単位で両話者を start 時刻統合し、直前 N 発話（話者問わず）を平均。
# 既定: N=1,2,3,4,5 を max_past=30秒 で（同一話者系列 run_exp_pastctx_same_series.sh と窓をそろえる）。
#   NS="1 2 3 4 5" max_past=30 ./run_exp_pastctx_session_series.sh   # 既定と同じ
#   NS="3 4 5"     max_past=30 ./run_exp_pastctx_session_series.sh   # 途中から
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export max_past=${max_past:-30}
for n in ${NS:-1 2 3 4 5}; do
    echo "===== session N=${n} (max_past=${max_past}s, 両話者) ====="
    context_scope=session N="${n}" max_past="${max_past}" ./run_ctxvec.sh "$@"
done
