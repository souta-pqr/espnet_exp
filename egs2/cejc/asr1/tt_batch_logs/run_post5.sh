#!/bin/bash
# 遅れの 90% 点を出すため，フレームダンプを「区間の長さ + 5 秒」に広げる（2026-10-16）。g・ge × eval_g・dev_g。
cd /home/kobori/2025/espnet/egs2/cejc/asr1
[ -e tt_batch_logs/post5.done ] && exit 0
[ -e tt_batch_logs/post5.running ] && exit 0
touch tt_batch_logs/post5.running
for s in -frame2-blk42xh2-ge -frame2-blk42xh2-g; do
  tag_suffix=$s post=5.0 sets="eval_g dev_g" ./run_frame_dump.sh
done
touch tt_batch_logs/post5.done
rm -f tt_batch_logs/post5.running
