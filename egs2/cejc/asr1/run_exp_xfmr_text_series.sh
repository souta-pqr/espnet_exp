#!/usr/bin/env bash
# 過去テキスト版を N=1..5 一括。
#   scope=same NS="1 2 3 4 5" pureasr_tag=20260713-pureasr ./run_exp_xfmr_text_series.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグ}; export scope=${scope:-same}; export max_past=${max_past:-30}
for n in ${NS:-1 2 3 4 5}; do
    echo "===================== text ${scope} N=${n} ====================="
    N="${n}" ./run_xfmr_text.sh "$@"
done
echo "===== 全 N 完了（text / ${scope}）====="
