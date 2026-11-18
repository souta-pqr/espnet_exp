#!/bin/bash
# abnt（silver13 で学習・ダンプ）が終わるのを待ち，ダンプを silver11 へ移して gm と同じ手順で評価する（2026-10-30）。
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
until ssh silver13 "test -e 2025/espnet/egs2/cejc/asr1/tt_batch_logs/ab_abnt_dump.done"; do
  ssh silver13 'pgrep -u kobori -fx "bash tt_batch_logs/run_ab.sh abnt"' > /dev/null || { echo "$(date +%m-%d\ %H:%M) abnt が止まった"; exit 1; }
  sleep 600
done
for ds in eval_g dev_g; do
  f=tt_preds/frame_attn-frame2-blk42xh2-abnt_same_n5_${ds}_valid.loss.ave_post5.0.tsv
  scp -q silver13:2025/espnet/egs2/cejc/asr1/$f $f && [ "$(ssh silver13 "wc -l < 2025/espnet/egs2/cejc/asr1/$f")" = "$(wc -l < $f)" ] \
    && ssh silver13 "rm 2025/espnet/egs2/cejc/asr1/$f" || { echo "移すのに失敗 $f"; exit 1; }
  echo "$(date +%m-%d\ %H:%M) $f を移した"
done
bash tt_analysis/longlat_20261024/run_eval_new.sh abnt || exit 1
nice -n 10 ~/.conda/envs/espnet/bin/python tt_analysis/longlat_20261024/budget.py gm abnp abnt abnl
