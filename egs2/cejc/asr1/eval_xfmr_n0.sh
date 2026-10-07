#!/usr/bin/env bash
# 真の N=0（凍結Enc＋Self-Attention・過去なし）のタグ性能を評価する。
#   ckpt=valid.tag_acc.ave.pth ./eval_xfmr_n0.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
ckpt=${ckpt:-valid.tag_acc.ave.pth}
tag_suffix=${tag_suffix:-}   # LoRA 版なら "-lora"
LOG=tt_eval_logs; mkdir -p "$LOG"

d=$(ls -d exp/asr_*-turntaking-xfmr${tag_suffix}-n0 2>/dev/null | tail -1 || true)
[ -n "$d" ] || { echo "N=0 の学習済みモデルが見つかりません。先に ./run_xfmr_n0.sh"; exit 1; }
[ -f "$d/$ckpt" ] || { echo "$d/$ckpt がありません"; exit 1; }

# past_vec.scp が残っていると評価側で誤って読まれないよう、--xfmr のみ（past_vec 未指定）で実行
out="${LOG}/n0${tag_suffix}_${ckpt%.pth}.log"
echo "===== eval N=0 (${d} / ${ckpt}) -> ${out} ====="
$PY local/turntaking/evaluate_multitask.py \
    --config "$d/config.yaml" --model "$d/$ckpt" \
    --data_dir dump/raw/eval --xfmr \
    > "$out" 2>&1

grep -E "macro-F1|<継続>|<終了>|<相槌>" "$out" | sed 's/^/    /'
