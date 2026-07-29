#!/usr/bin/env bash
# Stage2(xfmr: 凍結Enc＋Self-Attention) を N=1..5 で一括実行。
#   pureasr_tag=20260713-pureasr scope=same NS="1 2 3 4 5" max_past=30 ./run_exp_xfmr_series.sh
# 過去ベクトルの uvec は N に依存しないのでキャッシュ再利用（初回のみ全発話を符号化）。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export scope=${scope:-same}
export max_past=${max_past:-30}
for n in ${NS:-1 2 3 4 5}; do
    echo "===================== xfmr ${scope} N=${n} (max_past=${max_past}s) ====================="
    N="${n}" ./run_xfmr.sh "$@"
done
echo "===== 全 N 完了 ====="
