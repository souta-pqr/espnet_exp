#!/usr/bin/env bash
# 複数話者(session)・seqcat 方式（過去音声を連結）の N=1..5 系列。
#   pool_range=current NS="1 2 3 4 5" ./run_exp_pastctx_session_seqcat_series.sh   # 現発話のみ pool
#   pool_range=whole   NS="1 2 3 4 5" ./run_exp_pastctx_session_seqcat_series.sh   # 過去+現発話全体 pool
# session = reco から _IC* を除いたセッション単位で両話者を start 時刻統合し、直前N発話（話者問わず）を連結。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export max_past=${max_past:-30}
export pool_range=${pool_range:-current}
for n in ${NS:-1 2 3 4 5}; do
    echo "===== session(seqcat/${pool_range}) N=${n} (max_past=${max_past}s, 両話者) ====="
    scope=session N="${n}" max_past="${max_past}" ./run_seqcat.sh "$@"
done
