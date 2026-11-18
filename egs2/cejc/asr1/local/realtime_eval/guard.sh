#!/bin/bash
# 実時間評価の見張り：silver13 の空きメモリが 4 GB を切ったら評価を止める（学習を守る．2026-11-03）
# 評価は録音ごとに子プロセスで流すので，子も止める
while true; do
  P=$(pgrep -f "rt_eval.py.*nshard 4")
  [ -z "$P" ] && { echo "$(date +%T) 評価は全部終わった"; exit; }
  A=$(awk "/MemAvailable/{print int(\$2/1e6)}" /proc/meminfo)
  if [ "$A" -lt 4 ]; then for p in $P; do pkill -P $p; kill $p; done; echo "$(date +%T) 空き ${A}GB のため止めた: $P"; exit; fi
  sleep 30
done
