#!/usr/bin/env bash
# Stage2+Adapter の N=0（過去発話なし）。
#   pureasr_tag=20260713-pureasr ./run_xfmr_adapter_n0.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export base_cfg=myconf/train_asr_turntaking_xfmr_adapter.yaml
export tag_suffix=-adapter
exec ./run_xfmr_n0.sh "$@"
