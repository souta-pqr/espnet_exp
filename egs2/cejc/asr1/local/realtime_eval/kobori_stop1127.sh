#!/bin/bash
# 2026-11-25：silver13 の GPU は 11/27 は使えない（11/28 から）．11/26 23:50 に kobori の GPU の仕事を全部止める．
# 先に連鎖のスクリプト（kobori_chain*.sh）を止めて次を始めさせず，次に GPU を使っている kobori のプロセスを止める．
cd ~/2026/work/rt_eval_20261102
L=stop1127.log
END=$(date -d "2026-11-26 23:50" +%s)
echo "$(date '+%m/%d %T') 見張りを始めた．11/26 23:50 に止める" >> $L
while [ $(date +%s) -lt $END ]; do sleep 60; done
echo "$(date '+%m/%d %T') 止める" >> $L
for p in $(ps -u kobori -o pid=,args= | awk '$2=="bash" && $3 ~ /kobori_chain[0-9]+\.sh$/ {print $1}'); do
  echo "  連鎖 $p: $(ps -o args= -p $p)" >> $L; kill $p
done
sleep 2
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $p 2>/dev/null)" = "kobori" ] || continue
  echo "  GPU $p: $(ps -o args= -p $p | cut -c1-150)" >> $L
  pkill -P $p; kill $p
done
pkill -u kobori -f "[r]t_eval.py"
sleep 30
echo "$(date '+%m/%d %T') 止めた後の GPU の kobori のプロセス:" >> $L
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  [ "$(ps -o user= -p $p 2>/dev/null)" = "kobori" ] && { echo "  残り $p．kill -9" >> $L; kill -9 $p; }
done
echo "$(date '+%m/%d %T') 終わり" >> $L
