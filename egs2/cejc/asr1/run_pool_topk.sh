#!/usr/bin/env bash
# 【固定長・hard】重要度上位 M フレームだけを選んで残す（選択型の圧縮）。
#   出典: J. W. Rae, A. Potapenko, S. M. Jayakumar, C. Hillier, T. P. Lillicrap.
#         "Compressive Transformers for Long-Range Sequence Modelling." ICLR 2020. arXiv:1911.05507
#         （論文の compression function "most-used" に相当）
#   機構: 学習スコアラで上位 M フレームを選び、時系列順に **値そのまま** 残す。
#         選択は微分できないので、選ばれたフレームに sigmoid スコアを掛けて勾配を通す。
#   対照: 同じ M の query（soft）と比べると「混ぜる vs 選ぶ」の切り分けになる。
#
#   pureasr_tag=20260713-pureasr N=1 slots=8 ./run_pool_topk.sh
#   pureasr_tag=20260713-pureasr N=5 slots=8 ./run_pool_topk.sh
#   for m in 4 8 16; do pureasr_tag=... N=1 slots=$m ./run_pool_topk.sh; done   # M を振る
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export pool=topk
export scope=${scope:-same}
export N=${N:-1}
export max_past=${max_past:-30}
export slots=${slots:-8}            # スロット数 M（実効は min(M, Tp)）
variant="${scope}_n${N}_p${max_past}"

awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    echo "       この状態で学習すると過去が入らないまま学習されます。" >&2
    exit 1
}

echo "================================================================"
echo " topk（固定長・hard / Rae+ ICLR2020 \"most-used\"）"
echo " スロット数 M=${slots}  scope=${scope}  N=${N}  max_past=${max_past}s"
echo " Stage1=${pureasr_tag}  タグ=…-pool_topk_m${slots}-${variant}"
echo "================================================================"
echo "学習開始後の確認: grep -c past_speech exp/asr_*-pool_topk_m${slots}-${variant}/config.yaml が 0 なら過去が入っていない"

./run_xfmr_pool.sh "$@"

echo "===== 完了。評価: pool=topk slots=${slots} scope=${scope} N=${N} ./eval_xfmr_pool.sh ====="
