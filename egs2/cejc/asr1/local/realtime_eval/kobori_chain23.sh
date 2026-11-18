#!/bin/bash
# Easy Turn（CEJC で fine-tuning）の応答の決め方を，開発データを実際に流して選ぶための実走（2026-11-18．chain22 の後）．
# g1 0.6・g2 2.0・th2 0.40・S 3 は今の決め方（preset easy_turn_ft）と同じにし，th1 を動かす．GPU のサーバ 4 つ・4 並列（6 つでは CUDA のメモリが足りなかった）
D=~/2026/work/rt_eval_20261102; E=~/2026/work/exp-realtime-epd
until [ -e $D/chain22.done ]; do sleep 120; done
cd $E
for p in 8781 8782 8783 8784; do setsid nohup ~/anaconda3/envs/easyturn/bin/python tools/et_server.py --port $p > $D/log_etsrv_$p.txt 2>&1 < /dev/null & done
for p in 8781 8782 8783 8784; do until curl -s -o /dev/null -X POST --data-binary "" http://127.0.0.1:$p/transcribe; do sleep 10; done; done
echo "=== $(date) サーバが立った"
for th in 0.47 0.60 0.35 0.70; do
  O=dev_g_et_th$th
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  for i in 0 1 2 3; do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/dev_g/wav.scp --out $D/$O --shard $i --nshard 4 --model easy_turn_ft --et_url http://127.0.0.1:878$((i + 1)) --g1 0.6 --th1 $th --g2 2.0 --th2 0.4 --S 3 > $D/log_${O}_$i.txt 2>&1 &
  done
  wait; echo 終わり $(date) > $D/$O.done
done
pkill -f "tools/et_server.py --port 878"
echo 終わり $(date) > $D/chain23.done
