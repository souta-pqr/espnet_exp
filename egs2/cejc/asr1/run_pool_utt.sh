#!/usr/bin/env bash
# 【会話単位】発話境界を復元し、過去発話ごとに要約する（中間審査スライドの V_1..V_N 構造）。
#   出典: 直接の出典はない。固定率・固定長が「音響的に無意味な区切り」なのに対し、
#         会話の単位で区切ることに意味があるかを見る対照条件。
#         境界を教師なしで推定する一般解は CIF（L. Dong, B. Xu. ICASSP 2020. arXiv:1905.11235）。
#   機構: past_speech は連結波形なので符号化後は切れ目が失われる。past_bounds（各過去発話の
#         サンプル数）から累積比でフレーム境界を復元し、発話ごとに要約する。
#   出力: 有効発話数 × k スロット（k=1 なら V_1..V_N と同粒度）
#
#   pureasr_tag=20260713-pureasr N=3 ./run_pool_utt.sh              # k=1・発話内 max
#   pureasr_tag=20260713-pureasr N=5 uslots=2 ./run_pool_utt.sh     # 発話を 2 分割
#
# 発話内の要約（max / mean / attn）を変えるには
# myconf/train_asr_turntaking_xfmr_pool_utt.yaml の tt_compress_fn を編集する。
#
# 要件: past_bounds.scp（build_past_audio.py が past_speech と同時に生成）。無い場合は
#       run_xfmr_pool.sh が自動で作り直す。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export pool=utt
export scope=${scope:-same}
export N=${N:-3}
export max_past=${max_past:-30}
export uslots=${uslots:-1}          # 1 発話あたりスロット数 k
variant="${scope}_n${N}_p${max_past}"

awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    echo "       この状態で学習すると過去が入らないまま学習されます。" >&2
    exit 1
}
awk '/Stage 11: ASR Training/{s=1} s && /past_bounds\.scp,past_bounds,text_int/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_bounds の配線がありません（pool=utt には必須）。" >&2
    exit 1
}

echo "================================================================"
echo " utt（会話単位 / 出典なし・RQ4 の対照条件）"
echo " 1 発話あたり k=${uslots} スロット  scope=${scope}  N=${N}  max_past=${max_past}s"
echo " Stage1=${pureasr_tag}  タグ=…-pool_utt_k${uslots}-${variant}"
echo "================================================================"
echo "学習開始後の確認: grep -c past_bounds exp/asr_*-pool_utt_k${uslots}-${variant}/config.yaml が 0 なら境界が入っていない"

./run_xfmr_pool.sh "$@"

echo "===== 完了。評価: pool=utt uslots=${uslots} scope=${scope} N=${N} ./eval_xfmr_pool.sh ====="
