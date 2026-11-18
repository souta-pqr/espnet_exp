#!/usr/bin/env bash
# フレームダンプのやり直し（2026-10-16）。
#  - 範囲：区間の長さ + 2 秒（以前は区間の先頭 3 秒だけ）
#  - 過去の符号化を学習側（_encode_past）と同じにした evaluate_multitask.py を使う
# 旧ラベル（-g）と (e) ラベル（-ge）の判定器の両方。出力は tt_preds/*_post2.tsv。cron から起動。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
M=tt_batch_logs/frame_post2
for s in -frame2-blk42xh2-ge -frame2-blk42xh2-g; do
  echo "===== [$(date '+%m/%d %H:%M')] ${s}"
  tag_suffix=$s post=2.0 pool=attn scope=same N=5 max_past=30 sets="eval_g dev_g" ./run_frame_dump.sh
done
touch ${M}.done
echo "全部完了 $(date '+%m/%d %H:%M')"
