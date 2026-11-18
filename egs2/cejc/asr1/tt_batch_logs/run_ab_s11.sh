#!/usr/bin/env bash
# silver11 のアブレーション（2026-10-26）：gt のダンプが終わるのを待ってから abnp → abnt を順に流す（abna は silver13）（学習とダンプは並行させない）。
# 起動：tmux の外から setsid nohup bash tt_batch_logs/run_ab_s11.sh >> tt_batch_logs/ab_s11.log 2>&1 &
cd /home/kobori/2025/espnet/egs2/cejc/asr1
until [ -e tt_batch_logs/g_stage2gt_dump.done ]; do sleep 60; done
for ab in abnp abnt; do
  bash tt_batch_logs/run_ab.sh $ab >> tt_batch_logs/ab_$ab.log 2>&1 || { echo "[$(date '+%m-%d %H:%M')] $ab 失敗"; exit 1; }
  echo "[$(date '+%m-%d %H:%M')] $ab 完了"
done
