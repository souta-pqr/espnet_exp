#!/usr/bin/env bash
# 【既存手法・再実験】学習クエリ 1 本の attention pooling で過去を 1 本に要約する。
#   分類: 固定長・soft（Np = 1）。PastQueryPool の M=1・単一ヘッドと同じ機構。
#   位置づけ: query の M を振ったときの M=1 に対応する下限参照点。
#
#   pureasr_tag=20260713-pureasr N=5 ./run_pool_attn.sh
#   pureasr_tag=20260713-pureasr N=5 scope=session ./run_pool_attn.sh
#
# 以前の実験は asr.sh の Stage 11 に past_* の配線が無く、過去が入らないまま学習されていた。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}
export pool=attn
echo "=== attention pooling（固定長・soft / Np=1）  scope=${scope:-same} N=${N:-1}"
exec ./run_xfmr_pool.sh "$@"
