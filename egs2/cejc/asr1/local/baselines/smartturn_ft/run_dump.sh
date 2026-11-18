#!/bin/bash
# 学習し直した Smart Turn で dev_g・eval_g のダンプを作る（2026-10-29，silver13）。
# 録音を N 個に分けて並列（1 本 1.8 GB・GPU は小さい）．1 本のスレッドは 3（多いと CPU を取り合って遅くなった）。
# silver13 で eval_g，silver11 で dev_g を流す（10/29）。
#   SETS=eval_g N=4 setsid nohup bash tt_analysis/smartturn_ft_20261028/run_dump.sh > tt_analysis/smartturn_ft_20261028/out/run_dump_eval_g.log 2>&1 < /dev/null &
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
D=tt_analysis/smartturn_ft_20261028; N=${N:-4}
PY=/home/kobori/anaconda3/envs/llm/bin/python; [ -x $PY ] || PY=/home/kobori/.conda/envs/espnet/bin/python   # silver13・silver11
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
mkdir -p $D/out tt_preds
for ds in ${SETS:-eval_g dev_g}; do
  echo "=== $(date '+%m-%d %H:%M') $ds"
  for p in $(seq 0 $((N - 1))); do
    nice -n 10 $PY $D/smartft_dump.py data/$ds $D/out/smartft_$ds.part$p.tsv $p $N > $D/out/smartft_$ds.part$p.log 2>&1 &
  done
  wait
  for p in $(seq 0 $((N - 1))); do tail -1 $D/out/smartft_$ds.part$p.log | grep -q '^ok$' || { echo "失敗 $ds part$p"; exit 1; }; done
  cat $(for p in $(seq 0 $((N - 1))); do echo $D/out/smartft_$ds.part$p.tsv; done) > tt_preds/smart_turn_ft_${ds}_post5.0.tsv || { echo "連結に失敗 $ds"; exit 1; }
  rm -f $D/out/smartft_$ds.part*.tsv
  echo "$(date '+%m-%d %H:%M') $ds 完了 $(wc -l < tt_preds/smart_turn_ft_${ds}_post5.0.tsv) 行"
done
echo "すべて完了"
