#!/bin/bash
# アブレーションを gm と同じ手順で評価する（2026-10-29，silver11）。1 本ずつ。
#   abnp：ダンプ（silver11）が終わるのを待つ
#   abnl：ダンプは silver13 で作った → こちらへ移して silver13 の方は消す
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
until [ -e tt_batch_logs/ab_abnp_dump.done ]; do
  pgrep -u kobori -fx "bash tt_batch_logs/run_ab.sh abnp" > /dev/null || { echo "abnp のダンプが止まった"; exit 1; }
  sleep 120
done
bash tt_analysis/longlat_20261024/run_eval_new.sh abnp || exit 1
for ds in eval_g dev_g; do
  f=tt_preds/frame_attn-frame2-blk42xh2-abnl_same_n5_${ds}_valid.loss.ave_post5.0.tsv
  scp -q silver13:2025/espnet/egs2/cejc/asr1/$f $f && [ "$(ssh silver13 "wc -l < 2025/espnet/egs2/cejc/asr1/$f")" = "$(wc -l < $f)" ] \
    && ssh silver13 "rm 2025/espnet/egs2/cejc/asr1/$f" || { echo "移すのに失敗 $f"; exit 1; }
  echo "$(date +%H:%M) $f を移した"
done
bash tt_analysis/longlat_20261024/run_eval_new.sh abnl
