#!/usr/bin/env bash
# Phase 6 step1: BERT を 3 個追加学習してばらつきを測る＋アンサンブル部材を作る。
#
# text_ceiling.py には乱数シードの制御が無い（seed/manual_seed 未使用）ため、
# 同じ引数でも実行ごとに分類ヘッドの初期化とデータ順が変わる。
# Phase 5 の −0.0041 がばらつきの範囲かを切り分けるのが目的。
# 1 回約 5 分。重いデコードは Phase 5 で完了済み。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
DEC=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave

echo "===== [$(date '+%m-%d %H:%M')] BERT シード反復 開始 ====="
for s in 2 3 4; do
  OUT=exp/text_bert_asr_blk42sp_s$s
  if [ -s "$OUT/model.safetensors" ]; then echo "既存: $OUT"; continue; fi
  echo "----- [$(date '+%H:%M')] seed試行 $s"
  $PY local/turntaking/text_ceiling.py --epochs 2 \
      --train_text "$DEC/train_nodup_dec/text" \
      --dev_text   "$DEC/train_dev_dec/text" \
      --eval_text  "$DEC/eval/text" \
      --save "$OUT" --out tt_analysis/text_bert_blk42sp_s$s.json \
    && echo "  OK $OUT" || echo "  FAIL $s"
done
echo "===== [$(date '+%m-%d %H:%M')] 単体性能の一覧 ====="
for f in tt_analysis/text_bert_asr.json tt_analysis/text_bert_blk42sp.json tt_analysis/text_bert_blk42sp_s*.json; do
  [ -f "$f" ] && $PY -c "
import json,sys
d=json.load(open('$f'))
v=d.get('eval（ASR 認識結果）',{})
print(f\"{'$f'.split('/')[-1]:<34} eval(ASR) {v.get('macro_f1',float('nan')):.4f}\")
"
done
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
