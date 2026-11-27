#!/bin/bash
# 2026-11-26：フレームの平均（音声区間の全フレームの平均）をやめたモデルを学習し，実時間のスコアを書き出す．
#   POOL=1 ：最後のフレームの出力で判定する．
#   POOL=30：最後の 30 フレーム（1 フレーム 33 ms なので約 1 秒）の平均で判定する．
# 読み出しのほかは提案手法（espmk1gs．run_gs.sh）と同じ（train_gs・CTCW=1.0・MAXSEC=12・lr 5e-4・2 エポック・bs 4・acc 16）．
# 補助タスクの出力層も best_aux.pt に残す（後で CER を測るため）．コードは複製（encac_pool・exp-realtime-epd-pool）．
# Easy Turn の実走（chain29）が終わってから始める（計算の遅れを測っているので GPU を共有しない）．学習と書き出しは並行させない．
# 11/27 は GPU が使えない（kobori_stop1127.sh が 11/26 23:55 に止める）．11/28 にもう一度流せば，済んだものは飛ばして続きから．
D=~/2026/work/rt_eval_20261102; W=~/2026/work/encac_pool; R=~/2026/work/exp-realtime-epd-pool
A=~/2025/espnet/egs2/cejc/asr1; PY=~/anaconda3/envs/espnet/bin/python
until [ -e $D/chain29.done ]; do sleep 60; done
while pgrep -u kobori -f "rt_eval.py|et_server.py" > /dev/null; do sleep 30; done
echo "=== $(date) 始める．GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_ALLOC_CONF=expandable_segments:True ASR1=$A
for N in 1 30; do
  X=espmk1gsp$N; O=$A/exp/encac_${X}_20261126
  if [ ! -e $O.done ]; then
    echo "=== $(date) 学習 POOL=$N"
    (cd $W && POOL=$N MAXSEC=12 CTCW=1.0 $PY train_ac.py $O --lr 5e-4 --train train_gs --bs 4 --acc 16 > $O.log 2>&1) || { echo "=== $(date) 学習が止まった"; exit 1; }
    echo 終わり $(date) > $O.done
  fi
  if [ ! -e $R/weights/encac_$X.pt ]; then
    (cd $W && POOL=$N $PY export_encac.py $O/best.pt $R/weights/encac_$X.pt) || { echo "=== $(date) 書き出しの重みが作れない"; exit 1; }
    (cd $R && $PY tools/check_encac.py weights/encac_$X.pt)
  fi
  for d in dev_g eval_g; do
    [ -e $D/dump_${d}_$X.done ] && continue
    echo "=== $(date) スコアの書き出し $d $X"
    bash $D/kobori_rt_dump_pool.sh $d 6 $X weights/encac_$X.pt || { echo "=== $(date) 書き出しが止まった"; exit 1; }
  done
done
echo "終わり $(date)" > $D/chain30.done
