#!/usr/bin/env bash
# Stage2 + LoRA: 凍結エンコーダに LoRA を挿し、区間末タグ損失でエンコーダを軽く適応。
# 過去発話の扱い(same/session, N, max_past)は完全凍結版 run_xfmr.sh と同一。
#   pureasr_tag=20260713-pureasr scope=same    N=1 ./run_xfmr_lora.sh
#   pureasr_tag=20260713-pureasr scope=session N=1 ./run_xfmr_lora.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export base_cfg=myconf/train_asr_turntaking_xfmr_lora.yaml
export tag_suffix=-lora
exec ./run_xfmr.sh "$@"
