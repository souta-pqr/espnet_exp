#!/usr/bin/env bash
# Stage2+LoRA を N=1..5 で一括実行。same / session を scope で切替。
#   pureasr_tag=20260713-pureasr scope=same    NS="0 1 2 3 4 5" max_past=30 ./run_exp_xfmr_lora_series.sh
#   pureasr_tag=20260713-pureasr scope=session NS="0 1 2 3 4 5" max_past=30 ./run_exp_xfmr_lora_series.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export scope=${scope:-same}
export max_past=${max_past:-30}
for n in ${NS:-0 1 2 3 4 5}; do
    if [ "${n}" = "0" ]; then
        echo "===================== xfmr+LoRA N=0 (過去発話なし) ====================="
        ./run_xfmr_lora_n0.sh "$@"
    else
        echo "===================== xfmr+LoRA ${scope} N=${n} (max_past=${max_past}s) ====================="
        N="${n}" ./run_xfmr_lora.sh "$@"
    fi
done
echo "===== 全 N 完了（LoRA / ${scope}）====="
