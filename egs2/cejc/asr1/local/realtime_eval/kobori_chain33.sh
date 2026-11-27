#!/bin/bash
# 2026-11-27 準備，11/28 0:00 から GPU で流す（11/27 は GPU を使わない）．上から順に 1 本ずつ（学習と書き出しは並行させない）．
#  1. 提案手法（espmk1gs．全フレームの平均）を，補助タスクの出力層（best_aux.pt）も残す形で学び直し，CER を測る．
#     設定は元の run_gs10.sh と同じ（train_gs・CTCW=1.0・MAXSEC=12・lr 5e-4・2 エポック・bs 2・acc 32）．
#  2. エンコーダ固定のアブレーション（encac_espmk1gsfz_20261126．学習は済み）の実時間のスコアを書き出し，設定を選ぶ．
#  3. 音声認識（補助タスクの CTC）だけでエンコーダを学習する（train_ac.py --ctc_only）．区間末のラベルの学習が CER を下げるかの公平な基準．設定は 1 と同じ．
# 止まっても，もう一度流せば済んだものは飛ばして続きから．
# 11/27 昼：1 は silver10，3 は silver9 で先に流し始めた（出力名 _20261127）．終われば silver13 に写して .done を置くので，ここでは飛ばす．
# そのため 2（書き出し）を先に流す．
D=~/2026/work/rt_eval_20261102; W=~/2026/work/encac_pool; R=~/2026/work/exp-realtime-epd-pool
A=~/2025/espnet/egs2/cejc/asr1; PY=~/anaconda3/envs/espnet/bin/python
until [ "$(date +%Y%m%d)" -ge 20261128 ]; do sleep 60; done
echo "=== $(date) 始める．GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
export ENC=espnet:$A/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth PYTHONPATH=$HOME/2025/espnet PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ASR1=$A

train() {   # <出力> <train_ac.py への追加の引数>
  local O=$1; shift
  [ -e $O.done ] && { [ -e $W/hyp_eval_g_$(basename $O).tsv ] || (cd $W && OMP_NUM_THREADS=2 $PY cer_aux.py $O 0 $A/data/eval_g hyp_eval_g_$(basename $O).tsv); return 0; }
  echo "=== $(date) 学習 $(basename $O)"
  (cd $W && POOL=0 MAXSEC=12 CTCW=1.0 $PY train_ac.py $O --lr 5e-4 --train train_gs --bs 2 --acc 32 "$@" > $O.log 2>&1) || { echo "=== $(date) 学習が止まった"; exit 1; }
  echo 終わり $(date) > $O.done
  (cd $W && OMP_NUM_THREADS=2 $PY cer_aux.py $O 0 $A/data/eval_g hyp_eval_g_$(basename $O).tsv) || { echo "=== $(date) CER が測れない"; exit 1; }
}

# 2
X=espmk1gsfz
if [ ! -e $R/weights/encac_$X.pt ]; then
  (cd $W && POOL=0 $PY export_encac.py $A/exp/encac_${X}_20261126/best.pt $R/weights/encac_$X.pt) || { echo "=== $(date) 書き出しの重みが作れない"; exit 1; }
  (cd $R && $PY tools/check_encac.py weights/encac_$X.pt)
fi
for d in dev_g eval_g; do
  [ -e $D/dump_${d}_$X.done ] && continue
  echo "=== $(date) スコアの書き出し $d $X"
  bash $D/kobori_rt_dump_pool.sh $d 6 $X weights/encac_$X.pt || { echo "=== $(date) 書き出しが止まった"; exit 1; }
done
if [ ! -e $D/pick_$X.done ]; then   # CPU だけなので 3 の学習と並べてよい
  (cd ~/2026/work/from_silver10_1120/2026/work/rt_eval_20261102 && $PY rt_pick.py $A/data/dev_g $A/tt_analysis/vad_silero/dev_g $D/dump_dev_g_$X $A/data/eval_g $A/tt_analysis/vad_silero/eval_g $D/dump_eval_g_$X > $D/pick_$X.txt 2>&1 && echo 終わり > $D/pick_$X.done) &
fi

# 1
train $A/exp/encac_espmk1gsr_20261127

# 3
train $A/exp/encac_espmk1gsctc_20261127 --ctc_only
wait
echo "終わり $(date)" > $D/chain33.done
