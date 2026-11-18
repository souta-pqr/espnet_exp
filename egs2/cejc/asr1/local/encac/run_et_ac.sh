#!/bin/bash
# Easy Turn のアブレーション 2：音響だけ（エンコーダ＋平均＋線形層．論文の Whisper + Linear．2026-11-03，silver13）
# アブレーション 1（run_et_notext.sh）が終わってから，学習 → 開発・評価データのダンプ（区切りは NOEND で同じ）
cd ~/2026/work/easy_turn/cejc
A=~/2025/espnet/egs2/cejc/asr1
O=$A/exp/easyturn_ac_20261103
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f run_et_notext.sh > /dev/null; do sleep 300; done
[ -e $A/tt_preds/easyturn_notext_dump.done ] || { echo "アブレーション 1 のダンプがない"; exit 1; }
echo "=== $(date) 学習"
~/anaconda3/envs/easyturn/bin/python train_ac.py $O > $A/exp/easyturn_ac_20261103.log 2>&1 || { echo "学習 失敗"; exit 1; }
for d in dev_g eval_g; do
  echo "=== $(date) ダンプ $d"
  AC=1 NOEND=1 ~/anaconda3/envs/easyturn/bin/python et_dump.py $O/best.pt $d $A/tt_preds/easyturn_ac_${d}_post5.0_noend.tsv > $A/exp/easyturn_ac_dump_$d.log 2>&1 || { echo "ダンプ $d 失敗"; exit 1; }
done
touch $A/tt_preds/easyturn_ac_dump.done
echo "=== $(date) 完了"
