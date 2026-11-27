#!/bin/bash
# 2026-11-21：Easy Turn を評価データで流す（chain25 の後）．開発データ（chain23）で th1 0.35／0.47／0.60／0.70 → cut-in rate 7.5／7.0／6.6／6.2%．
# 提案手法の上限（≤ 8%・≤ 7%・≤ 6%）に合わせて th1 0.35・0.47・0.70 を，0.47 → 0.35 → 0.70 の順に流す．ほかは chain23 と同じ
D=~/2026/work/rt_eval_20261102; E=~/2026/work/exp-realtime-epd
until [ -e $D/chain25.done ]; do sleep 120; done
cd $E
for p in 8781 8782 8783 8784; do setsid nohup ~/anaconda3/envs/easyturn/bin/python tools/et_server.py --port $p > $D/log_etsrv_$p.txt 2>&1 < /dev/null & done
for p in 8781 8782 8783 8784; do until curl -s -o /dev/null -X POST --data-binary "" http://127.0.0.1:$p/transcribe; do sleep 10; done; done
for th in 0.47 0.35 0.70; do
  O=eval_g_et_th$th
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  pids=()
  for i in 0 1 2 3; do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/eval_g/wav.scp --out $D/$O --shard $i --nshard 4 --model easy_turn_ft --et_url http://127.0.0.1:878$((i + 1)) --g1 0.6 --th1 $th --g2 2.0 --th2 0.4 --S 3 > $D/log_${O}_$i.txt 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"; echo 終わり $(date) > $D/$O.done
done
pkill -f "tools/et_server.py --port 878"
echo 終わり $(date) > $D/chain26.done
