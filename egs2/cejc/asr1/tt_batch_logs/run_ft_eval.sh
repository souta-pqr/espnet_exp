#!/bin/bash
# 学習し直した Smart Turn・VAP を評価する（2026-10-29，silver11）。ダンプがそろうのを待って 1 本ずつ流す。
#   smartft：評価データのダンプは silver13 で作る → こちらへ移して silver13 の方は消す
#   vapft：両方ともこちらで作る（tt_analysis/vap_ft_20261029/out/dump.log）
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
f=tt_preds/smart_turn_ft_eval_g_post5.0.tsv
until ssh silver13 "grep -q すべて完了 ~/2025/espnet/egs2/cejc/asr1/tt_analysis/smartturn_ft_20261028/out/run_dump_eval_g.log && test -s ~/2025/espnet/egs2/cejc/asr1/$f"; do sleep 120; done
scp -q silver13:2025/espnet/egs2/cejc/asr1/$f $f && cmp -s <(ssh silver13 "wc -l < 2025/espnet/egs2/cejc/asr1/$f") <(wc -l < $f) \
  && ssh silver13 "rm 2025/espnet/egs2/cejc/asr1/$f" || { echo "移すのに失敗"; exit 1; }
echo "$(date +%H:%M) $f を移した（$(wc -l < $f) 行）"
bash tt_analysis/longlat_20261024/run_eval_new.sh smartft || exit 1
until grep -q すべて完了 tt_analysis/vap_ft_20261029/out/dump.log; do sleep 120; done
bash tt_analysis/longlat_20261024/run_eval_new.sh vapft
