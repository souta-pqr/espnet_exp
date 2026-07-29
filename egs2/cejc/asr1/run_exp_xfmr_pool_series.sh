#!/usr/bin/env bash
# 過去発話プーリング比較（max / attn）を N=1..5 で一括実行。same / session を scope で切替。
#   pool=attn scope=same    NS="1 2 3 4 5" max_past=30 pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_series.sh
#   pool=attn scope=session NS="1 2 3 4 5" max_past=30 pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_series.sh
#   pool=max  ... （比較対照。同じデータ経路で max pooling）
#   pool=conv  ratio=8 scope=same NS="1 2 3 4 5" ... （固定率・soft。ratio で圧縮率を振る）
#   pool=query slots=8 scope=same NS="1 2 3 4 5" ... （固定長・soft。slots でスロット数を振る）
#   pool=topk  slots=8 scope=same NS="1 2 3 4 5" ... （固定長・hard。slots でスロット数を振る）
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pool=${pool:?pool=max または pool=attn を指定}
export pureasr_tag=${pureasr_tag:?Stage1のタグ 例) pureasr_tag=20260713-pureasr}
export scope=${scope:-same}
export max_past=${max_past:-30}
export ratio=${ratio:-8}        # pool=conv の圧縮率
export slots=${slots:-8}        # pool=query / topk のスロット数
for n in ${NS:-1 2 3 4 5}; do
    echo "===================== pool=${pool} ${scope} N=${n} (max_past=${max_past}s) ====================="
    N="${n}" ./run_xfmr_pool.sh "$@"
done
echo "===== 全 N 完了（pool=${pool} / ${scope}）====="
