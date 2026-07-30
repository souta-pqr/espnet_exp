#!/usr/bin/env bash
# 【固定長・soft】学習クエリ M 本で過去を要約する（PMA / Perceiver 型）。
#   出典: J. Lee, Y. Lee, J. Kim, A. R. Kosiorek, S. Choi, Y. W. Teh.
#         "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks."
#         ICML 2019. arXiv:1810.00825
#         A. Jaegle, F. Gimeno, A. Brock, A. Zisserman, O. Vinyals, J. Carreira.
#         "Perceiver: General Perception with Iterative Attention." ICML 2021. arXiv:2103.03206
#   機構: 学習された M 本のクエリが過去フレーム列に cross-attention（4 head）→ 常に M スロット
#   端点: M=1・単一ヘッド → 既存の attention pooling と同じ
#
#   pureasr_tag=20260713-pureasr N=1 slots=8 ./run_pool_query.sh
#   pureasr_tag=20260713-pureasr N=5 slots=8 ./run_pool_query.sh
#   for m in 1 2 4 8 16; do pureasr_tag=... N=1 slots=$m ./run_pool_query.sh; done   # M を振る
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export pool=query
export scope=${scope:-same}
export N=${N:-1}
export max_past=${max_past:-30}
export slots=${slots:-8}            # スロット数 M（過去長に依らず固定）
variant="${scope}_n${N}_p${max_past}"

awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    echo "       この状態で学習すると過去が入らないまま学習されます。" >&2
    exit 1
}

echo "================================================================"
echo " query（固定長・soft / Lee+ ICML2019, Jaegle+ ICML2021）"
echo " スロット数 M=${slots}  scope=${scope}  N=${N}  max_past=${max_past}s"
echo " Stage1=${pureasr_tag}  タグ=…-pool_query_m${slots}-${variant}"
echo "================================================================"
echo "学習開始後の確認: grep -c past_speech exp/asr_*-pool_query_m${slots}-${variant}/config.yaml が 0 なら過去が入っていない"

./run_xfmr_pool.sh "$@"

echo "===== 完了。評価: pool=query slots=${slots} scope=${scope} N=${N} ./eval_xfmr_pool.sh ====="
