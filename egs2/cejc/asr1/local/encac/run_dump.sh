#!/bin/bash
cd ~/2026/work/easy_turn/cejc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=~/2025/espnet/egs2/cejc/asr1
for d in dev_g eval_g; do
  ~/anaconda3/envs/easyturn/bin/python et_dump.py $A/exp/easyturn_ft_20261101/best.pt $d $A/tt_preds/easyturn_ft_${d}_post5.0.tsv > $A/exp/easyturn_dump_$d.log 2>&1 || exit 1
done
touch $A/tt_preds/easyturn_ft_dump.done
