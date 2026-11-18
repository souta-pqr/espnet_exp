#!/bin/bash
# 新しく学習した判定器（gt・アブレーション）を gm と同じ手順で評価する（2026-10-26，silver11）。
#   1. 開発データ：1 段の規則の曲線（eot_eval.py．候補は run_cmp_s11.sh と同じ）→ newlabel_20261021_c8/dev に集める
#   2. 開発データ：2 段の規則の曲線（two_stage.py）→ curve_2stage_<名前>.tsv
#   3. 評価データ：全方式に同じ選び方 A・B（eval_all.py）→ pick_all_<名前>.txt（<名前> − 各方式の差つき）
# どの段もメモリの上限 16 GB（ulimit -v），nice 10．1 本ずつ流す。
#   bash tt_analysis/longlat_20261024/run_eval_new.sh gt
set -u
n=$1
cd /home/kobori/2025/espnet/egs2/cejc/asr1
PY=~/.conda/envs/espnet/bin/python
OUT=tt_analysis/newlabel_20261021_c8; L=tt_analysis/longlat_20261024
#   CEJC で学習し直した従来手法（smartft・vapft，10/29）も同じ手順で流せる（先読み 0）
dump() { case $n in smartft) echo "tt_preds/smart_turn_ft_$1_post5.0.tsv";; vapft) echo "tt_preds/vap_ft_$1_post5.0.tsv";;
                    *) echo "tt_preds/frame_attn-frame2-blk42xh2-${n}_same_n5_$1_valid.loss.ave_post5.0.tsv";; esac; }
case $n in smartft|vapft) LA=0;; *) LA=0.2;; esac
run() { ( ulimit -v 16000000; nice -n 10 /usr/bin/time -f "最大メモリ %M KB  %e 秒" "$@" ); }
for ds in dev_g eval_g; do [ -s "$(dump $ds)" ] || { echo "ダンプがない: $(dump $ds)"; exit 1; }; done

if [ ! -s $OUT/dev_$n/curve_$n+silero+fb.tsv ]; then
  echo "=== $(date +%H:%M) 1. 開発データ（1 段の規則）$n"
  run $PY local/turntaking/eot_eval.py --data data/dev_g --dump $(dump dev_g) --name $n --lookaheads $LA \
      --silero tt_analysis/vad_silero/dev_g --sil_max 3.0 --window 5 --free_sec 2 --sustain 0 \
      --gates "0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.8 1.0 1.2 1.5" --fb_gates "0.6 0.8 1.0 1.2 1.5" \
      --fallback "1.0 1.5 2.0 2.5 3.0" --fb_bc "1.01 0.3 0.1" \
      --out $OUT/dev_$n > $OUT/dev_$n.log 2> $OUT/dev_$n.err || { echo "失敗"; tail -3 $OUT/dev_$n.err; exit 1; }
  grep "最大メモリ" $OUT/dev_$n.err
fi
cp $OUT/dev_$n/curve_$n+silero.tsv $OUT/dev_$n/curve_$n+silero+fb.tsv $OUT/dev/

if [ ! -s $L/curve_2stage_$n.tsv ]; then
  echo "=== $(date +%H:%M) 2. 開発データ（2 段の規則）$n"
  LA=$LA TAG=_$n run $PY $L/two_stage.py dev data/dev_g $(dump dev_g) $L > $L/two_stage_$n.log 2>&1 \
    || { echo "失敗"; tail -3 $L/two_stage_$n.log; exit 1; }
  grep "最大メモリ" $L/two_stage_$n.log
fi

echo "=== $(date +%H:%M) 3. 評価データ（全方式・選び方 A・B）$n"
run $PY $L/eval_all.py $n > $L/eval_all_$n.log 2>&1 || { echo "失敗"; tail -3 $L/eval_all_$n.log; exit 1; }
grep "最大メモリ" $L/eval_all_$n.log
cat $L/pick_all_$n.txt
echo "=== $(date +%H:%M) 完了"
