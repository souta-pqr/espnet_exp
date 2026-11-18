#!/usr/bin/env bash
# silver13 のアブレーション（2026-10-27）：abnl（学習とダンプ）が終わるのを待ってから abnt を流す。
#   silver11 はディスクが混んで 1 エポック 5 時間かかるため，abnt を silver11 から移した。
# 起動：setsid nohup bash tt_batch_logs/run_ab_s13.sh >> tt_batch_logs/ab_s13.log 2>&1 < /dev/null &
cd /home/kobori/2025/espnet/egs2/cejc/asr1
until [ -e tt_batch_logs/ab_abnl_dump.done ]; do
  pgrep -u kobori -fx "bash tt_batch_logs/run_ab.sh abnl" > /dev/null || { echo "[$(date '+%m-%d %H:%M')] abnl が止まったので abnt は始めない"; exit 1; }
  sleep 300
done
echo "[$(date '+%m-%d %H:%M')] abnl 完了．abnt を始める"
bash tt_batch_logs/run_ab.sh abnt >> tt_batch_logs/ab_abnt.log 2>&1 || { echo "[$(date '+%m-%d %H:%M')] abnt 失敗"; exit 1; }
echo "[$(date '+%m-%d %H:%M')] abnt 完了"
