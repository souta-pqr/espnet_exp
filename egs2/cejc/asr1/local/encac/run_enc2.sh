#!/bin/bash
# エンコーダを取り替えた音響だけのモデルを順に流す（2026-11-04，silver13）．run_enc.sh esp が終わってから始める
# 1 本ごとに：学習 → 開発・評価データのダンプ（NOEND）．メモリが足りなければ 1 回の量を 1/4・勾配をためる回数を 4 倍にしてやり直す
cd ~/2026/work/easy_turn/cejc
A=~/2025/espnet/egs2/cejc/asr1
PY=~/anaconda3/envs/easyturn/bin/python
export PYTHONPATH=$HOME/2026/work/easy_turn/pylib_tf446 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "run_enc.sh esp" > /dev/null; do sleep 300; done
one() {   # <名前> <ENC> <lr>
  N=$1; export ENC=$2; O=$A/exp/encac_${N}_20261104
  [ -e $A/tt_preds/encac_${N}_dump.done ] && return 0
  echo "=== $(date) 学習 $N $ENC lr $3"
  if ! $PY train_ac.py $O --lr $3 > $O.log 2>&1; then
    grep -q OutOfMemoryError $O.log || { echo "学習 失敗 $N"; return 1; }
    echo "=== $(date) メモリ不足．1 回 4・ためる 16 でやり直す"
    mv $O.log $O.oom.log
    $PY train_ac.py $O --lr $3 --bs 4 --acc 16 > $O.log 2>&1 || { echo "学習 失敗 $N"; return 1; }
  fi
  for d in dev_g eval_g; do
    echo "=== $(date) ダンプ $N $d"
    AC=1 NOEND=1 $PY et_dump.py $O/best.pt $d $A/tt_preds/encac_${N}_${d}_post5.0_noend.tsv > $A/exp/encac_${N}_dump_$d.log 2>&1 || { echo "ダンプ 失敗 $N $d"; return 1; }
  done
  touch $A/tt_preds/encac_${N}_dump.done
}
one mimi mimi:kyutai/mimi 5e-5
one wmedc whisper:openai/whisper-medium:chunk=25 5e-5
one wmed whisper:openai/whisper-medium 5e-5
one w2vl w2v:reazon-research/japanese-wav2vec2-large-rs35kh 5e-5
one wsmall whisper:openai/whisper-small 5e-5
echo "=== $(date) 全部終わり"
