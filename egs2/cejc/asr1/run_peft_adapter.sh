#!/usr/bin/env bash
# 【PEFT・再実験】各 Conformer ブロックの FFN 出力に Bottleneck Adapter を加算して適応する。
#   出典: N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe,
#         A. Gesmundo, M. Attariyan, S. Gelly.
#         "Parameter-Efficient Transfer Learning for NLP." ICML 2019. arXiv:1902.00751
#   設定: bottleneck=32, 主 FFN のみ 12 箇所（Pfeiffer 型）, up を零初期化,
#         補正の大きさ <= 0.4 × 入力の大きさ（N=0 の掃引で選定）。
#   過去要約は pool= で選ぶ（既定 attn）。PEFT 以外の条件は pool 系と共通。
#   注意: エンコーダが適応するため CER は Stage1 と同一ではなくなる（劣化量も一緒に見る）。
#
#   pureasr_tag=20260713-pureasr N=5 ./run_peft_adapter.sh
#   pureasr_tag=20260713-pureasr N=5 pool=query slots=8 ./run_peft_adapter.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}
export pool=${pool:-attn}
export base_cfg=myconf/train_asr_turntaking_xfmr_peft_adapter.yaml
export tag_suffix=-adapter
echo "=== Adapter（bottleneck=32 / max_ratio=0.4）＋ 過去要約=${pool}  scope=${scope:-same} N=${N:-1}"
exec ./run_xfmr_pool.sh "$@"
