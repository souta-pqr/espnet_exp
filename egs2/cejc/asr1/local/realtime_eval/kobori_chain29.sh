#!/bin/bash
# 2026-11-24：Easy Turn の上限 6% を仕上げる（chain23・26 の続き）．
# ユーザ：silver13 の GPU は 11/25 0:00 から使ってよい（予約してある）．それまでは何も流さない（このスクリプトは時刻まで眠るだけ）．
# ほかの人のプロセスが GPU に残っていても始める（ユーザ）．計算の遅れに響くので，残っていたかを chain29_gpu.log に書く．
# 開発データで th1 0.70 → cut-in rate 6.2% だった．0.75 を流し，6.0% を超えたら 0.80 も流す．
# 開発データで 6.0% 以下になった閾値（試した中で最も小さいもの＝latency が最も短いもの）で評価データを流す．ほかの値は chain23 と同じ．
D=~/2026/work/rt_eval_20261102; E=~/2026/work/exp-realtime-epd
A=~/2025/espnet/egs2/cejc/asr1; F=~/2026/work/from_silver10_1120/2026/work/rt_eval_20261102
PY=~/anaconda3/envs/espnet/bin/python
START=$(date -d "2026-11-25 00:00" +%s)  # 過ぎていればすぐ次へ
until [ "$(date +%s)" -ge "$START" ]; do sleep 60; done
gpu_others() {  # GPU を使っている kobori 以外のプロセス
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    u=$(ps -o user= -p $p | tr -d " "); [ -n "$u" ] && [ "$u" != kobori ] && echo -n " $u:$p"
  done
}
echo "=== $(date) 始める．GPU のほかの人のプロセス:$(gpu_others)"
(while true; do echo "$(date "+%m/%d %T") 使用率 $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)，ほかの人:$(gpu_others)"; sleep 600; done) > $D/chain29_gpu.log 2>&1 &
WATCH=$!
# 11/25 0:00 に始めたところ，ほかの人のプロセスが GPU を 22 GB 使っていて，Easy Turn のサーバ 4 つ（1 つ約 2.4 GB）が CUDA のメモリ不足で落ちた．
# → GPU の空きが 11 GB 以上になるまで待ってからサーバを立てる．
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 11000 ]; do sleep 300; done
echo "=== $(date) GPU の空きが 11 GB を超えた．GPU のほかの人のプロセス:$(gpu_others)"
cd $E
for p in 8781 8782 8783 8784; do setsid nohup ~/anaconda3/envs/easyturn/bin/python tools/et_server.py --port $p > $D/log_etsrv_$p.txt 2>&1 < /dev/null & done
for p in 8781 8782 8783 8784; do until curl -s -o /dev/null -X POST --data-binary "" http://127.0.0.1:$p/transcribe; do pgrep -f "et_server.py --port $p" > /dev/null || { echo "=== $(date) サーバ $p が落ちた．止める"; pkill -f "tools/et_server.py --port 878"; kill $WATCH; exit 1; }; sleep 10; done; done
echo "=== $(date) サーバが立った"
run() {  # run dev_g|eval_g 閾値
  local d=$1 th=$2 O=$1_et_th$2 pids=()
  [ -e $D/$O.done ] && return
  echo "=== $(date) $O"
  for i in 0 1 2 3; do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp $A/data/$d/wav.scp --out $D/$O --shard $i --nshard 4 --model easy_turn_ft --et_url http://127.0.0.1:878$((i + 1)) --g1 0.6 --th1 $th --g2 2.0 --th2 0.4 --S 3 > $D/log_${O}_$i.txt 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"; echo 終わり $(date) > $D/$O.done
}
cutin() {  # 開発データでの cut-in rate (%)
  (cd $D && $PY rt_table.py $A/data/dev_g $A/tt_analysis/vad_silero/dev_g $F/dump_dev_g_espmk1gs x=$D/dev_g_et_th$1 | tail -1 | cut -f2 | tr -d %)
}
TH=""
for th in 0.75 0.80; do
  run dev_g $th
  c=$(cutin $th); echo "=== 開発データ th1 $th：cut-in rate $c%"
  if awk "BEGIN{exit !($c <= 6.0)}"; then TH=$th; break; fi
done
if [ -n "$TH" ]; then run eval_g $TH; else echo "=== 0.80 でも 6% を超えた．評価データは流さない"; fi
pkill -f "tools/et_server.py --port 878"
kill $WATCH
echo "終わり $(date) th1=$TH" > $D/chain29.done
