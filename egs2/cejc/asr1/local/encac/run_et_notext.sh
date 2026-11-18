#!/bin/bash
# Easy Turn のアブレーション 1：書き起こしを出さずにタグだけを出す（論文の Easy Turn-only-state．2026-11-02，silver13）
# 種 31 の gt（run_gtseed.sh）が終わってから，学習 → 開発・評価データのダンプ（区間末の時刻は使わない＝NOEND）
cd ~/2026/work/easy_turn/cejc
A=~/2025/espnet/egs2/cejc/asr1
O=$A/exp/easyturn_ft_notext_20261103
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f run_gtseed.sh > /dev/null; do sleep 300; done
[ -e $A/tt_batch_logs/ab_gts31_dump.done ] || { echo "種 31 のダンプがない"; exit 1; }
echo "=== $(date) 学習"
~/anaconda3/envs/easyturn/bin/python train_et.py $O --no_text > $A/exp/easyturn_ft_notext_20261103.log 2>&1 || { echo "学習 失敗"; exit 1; }
for d in dev_g eval_g; do
  echo "=== $(date) ダンプ $d"
  NOEND=1 NOTEXT=1 ~/anaconda3/envs/easyturn/bin/python et_dump.py $O/best.pt $d $A/tt_preds/easyturn_notext_${d}_post5.0_noend.tsv > $A/exp/easyturn_notext_dump_$d.log 2>&1 || { echo "ダンプ $d 失敗"; exit 1; }
done
touch $A/tt_preds/easyturn_notext_dump.done
echo "=== $(date) 完了"
