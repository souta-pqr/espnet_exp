#!/usr/bin/env bash
# 【PEFT・再実験】凍結エンコーダに LoRA を挿し、区間末タグ損失でエンコーダを軽く適応する。
#   出典: E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen.
#         "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685
#   設定: rank=8, alpha=8, dropout=0.05, 各 Conformer ブロックの自己注意 Q,V に挿入。
#   過去要約は pool= で選ぶ（既定 attn）。PEFT 以外の条件は pool 系と共通。
#   注意: エンコーダが適応するため CER は Stage1 と同一ではなくなる（劣化量も一緒に見る）。
#
#   pureasr_tag=20260713-pureasr N=5 ./run_peft_lora.sh
#   pureasr_tag=20260713-pureasr N=5 pool=query slots=8 ./run_peft_lora.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}
export pool=${pool:-attn}
export base_cfg=myconf/train_asr_turntaking_xfmr_peft_lora.yaml
export tag_suffix=-lora
echo "=== LoRA（rank=8 / Q,V）＋ 過去要約=${pool}  scope=${scope:-same} N=${N:-1}"
exec ./run_xfmr_pool.sh "$@"
