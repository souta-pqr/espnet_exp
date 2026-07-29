#!/usr/bin/env bash
# 同一話者・生特徴量(エンコーダ非経由の log-mel fbank 平均)版 N系列。ASR不変・未来なし。
# ctx_vec = 直前N発話の「生fbackを時間平均した80次元」を head に concat（encoder出力は使わない）。
#   NS="1 2 3 4 5" max_past=30 ./run_exp_pastctx_same_raw_series.sh
#   feat_norm=none で global_mvn なしの完全な生log-mel
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export feat_source=raw
export feat_norm=${feat_norm:-global}
export max_past=${max_past:-30}
for n in ${NS:-1 2 3 4 5}; do
    echo "===== same(raw fbank) N=${n} (max_past=${max_past}s, feat_norm=${feat_norm}) ====="
    context_scope=same N="${n}" max_past="${max_past}" ./run_ctxvec.sh "$@"
done
