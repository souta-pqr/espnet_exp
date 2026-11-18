#!/bin/bash
# silver11 で gt 版アブレーション（ASR の事前学習なし）を流す（2026-10-30）：10/31 になってから。過去の文脈なしは silver12 で流す。
# ASR の事前学習なしの初期値（gm 版．silver13 で学習）は始める前に silver13 から持ってきて，学習が終わったら消す。
# 起動：setsid nohup bash tt_batch_logs/run_abgt_silver11.sh >> tt_batch_logs/abgt_silver11.log 2>&1 < /dev/null &
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
until [ "$(date +%Y%m%d)" -ge 20261031 ]; do sleep 600; done
d=exp/asr_20261026-turntaking-xfmr-pool_attn-frame2-blk42xh2-abnl-same_n5_p30
mkdir -p $d
scp -q silver13:2025/espnet/egs2/cejc/asr1/$d/valid.loss.ave_10best.pth $d/ && ln -sf valid.loss.ave_10best.pth $d/valid.loss.ave.pth || { echo "初期値を持ってこられない"; exit 1; }
NEED_MB=22000 bash tt_batch_logs/run_abgt.sh gtnoinit >> tt_batch_logs/ab_gtnoinit.log 2>&1 || { echo "$(date +%m-%d\ %H:%M) ASR の事前学習なしで失敗"; exit 1; }
rm -r $d
echo "$(date +%m-%d\ %H:%M) すべて完了"
