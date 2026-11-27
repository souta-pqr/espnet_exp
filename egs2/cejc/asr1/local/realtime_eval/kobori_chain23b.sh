#!/bin/bash
# chain23 の続き（2026-11-20）．chain23 は Easy Turn のサーバもバックグラウンドで立てていたので，実走の後の wait がサーバの終わりまで待ち，
# 1 本目（th1 0.47．11/19 17:26 に 163 本とも終わっていた）の後で止まっていた．実走のプロセスだけを待つように直して，残りの 3 本を流す．
# サーバは chain23 が立てたもの（8781〜8784）をそのまま使う．ほかは chain23 と同じ
D=~/2026/work/rt_eval_20261102; E=~/2026/work/exp-realtime-epd
cd $E
for p in 8781 8782 8783 8784; do until curl -s -o /dev/null -X POST --data-binary "" http://127.0.0.1:$p/transcribe; do sleep 10; done; done
for th in 0.47 0.60 0.35 0.70; do
  O=dev_g_et_th$th
  [ -e $D/$O.done ] && continue
  echo "=== $(date) $O"
  pids=()
  for i in 0 1 2 3; do
    OMP_NUM_THREADS=2 ~/anaconda3/envs/epd/bin/python $D/rt_eval.py --wav_scp ~/2025/espnet/egs2/cejc/asr1/data/dev_g/wav.scp --out $D/$O --shard $i --nshard 4 --model easy_turn_ft --et_url http://127.0.0.1:878$((i + 1)) --g1 0.6 --th1 $th --g2 2.0 --th2 0.4 --S 3 > $D/log_${O}_$i.txt 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"; echo 終わり $(date) > $D/$O.done
done
pkill -f "tools/et_server.py --port 878"
echo 終わり $(date) > $D/chain23.done
