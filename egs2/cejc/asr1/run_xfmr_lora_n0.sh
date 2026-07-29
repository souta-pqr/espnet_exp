#!/usr/bin/env bash
# Stage2+LoRA の N=0（過去発話なし）。LoRA 系列の左端を測るための条件。
#   pureasr_tag=20260713-pureasr ./run_xfmr_lora_n0.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export base_cfg=myconf/train_asr_turntaking_xfmr_lora.yaml
export tag_suffix=-lora
exec ./run_xfmr_n0.sh "$@"
