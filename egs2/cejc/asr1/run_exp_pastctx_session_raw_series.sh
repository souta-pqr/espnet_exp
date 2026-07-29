#!/usr/bin/env bash
# 複数話者(session)・生特徴量(エンコーダ非経由の log-mel fbank 平均)版 N系列。ASR不変・未来なし。
#   NS="1 2 3 4 5" max_past=30 ./run_exp_pastctx_session_raw_series.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export feat_source=raw
export feat_norm=${feat_norm:-global}
export max_past=${max_past:-30}
for n in ${NS:-1 2 3 4 5}; do
    echo "===== session(raw fbank) N=${n} (max_past=${max_past}s, 両話者, feat_norm=${feat_norm}) ====="
    context_scope=session N="${n}" max_past="${max_past}" ./run_ctxvec.sh "$@"
done
