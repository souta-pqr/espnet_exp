#!/usr/bin/env bash
# 【既存手法・再実験】過去音声を凍結エンコーダで符号化し max pooling で 1 本に潰す。
#   分類: 固定長・非学習（Np = 1）。学習クエリを持たないので query の M=1 より弱い下限。
#   位置づけ: conv の c>=Tp と数値的に一致する端点。比較の下限参照点。
#
#   pureasr_tag=20260713-pureasr N=5 ./run_pool_max.sh
#   pureasr_tag=20260713-pureasr N=5 scope=session ./run_pool_max.sh
#
# 以前の実験は asr.sh の Stage 11 に past_* の配線が無く、過去が入らないまま学習されていた。
# 配線が入っているかは run_xfmr_pool.sh 経由で下の検査が担保する。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}
export pool=max
echo "=== max pooling（固定長・非学習 / Np=1）  scope=${scope:-same} N=${N:-1}"
exec ./run_xfmr_pool.sh "$@"
