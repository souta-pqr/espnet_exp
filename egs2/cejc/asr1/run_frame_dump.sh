#!/usr/bin/env bash
# フレーム単位ダンプ：1 回の符号化で全時点の判定を出す。
#   tag_suffix=-trunc-new ./run_frame_dump.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
pool=${pool:-attn}; scope=${scope:-same}; N=${N:-5}; max_past=${max_past:-30}
tag_suffix=${tag_suffix:--trunc-new}; ckpt=${ckpt:-valid.loss.ave.pth}
maxsec=${maxsec:-3.0}; sets=${sets:-"eval train_dev"}
d=$(ls -d exp/asr_*-turntaking-xfmr-pool_${pool}${tag_suffix}-${scope}_n${N}_p${max_past} 2>/dev/null | tail -1)
[ -n "$d" ] || { echo "モデルが見つかりません"; exit 1; }
mkdir -p tt_preds tt_eval_logs
for ds in ${sets}; do
  out="tt_preds/frame_${pool}${tag_suffix}_${scope}_n${N}_${ds}_${ckpt%.pth}.tsv"
  [ -s "$out" ] && { echo "既存: $out"; continue; }
  echo "----- [$(date +%H:%M:%S)] ${ds}"
  $PY local/turntaking/evaluate_multitask.py \
      --config "$d/config.yaml" --model "$d/$ckpt" \
      --data_dir "dump/raw/${ds}" --past_pool \
      --past_speech_scp "dump/raw/${ds}/past_speech_${scope}_n${N}_p${max_past}.scp" \
      --frame_level --frame_max_sec "${maxsec}" \
      --full_wav_scp "data/${ds}/wav.scp" --segments "data/${ds}/segments" \
      --dump_preds "$out" > "tt_eval_logs/frame_${tag_suffix}_${ds}.log" 2>&1 \
      && echo "  OK $(wc -l < "$out") 行" || echo "  FAIL"
done
echo "完了 $(date +%H:%M:%S)"
