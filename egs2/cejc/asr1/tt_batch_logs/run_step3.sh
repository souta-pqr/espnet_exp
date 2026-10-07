#!/usr/bin/env bash
# Phase 6 step3: 「学習テキストが汚い方が頑健」という step2 の知見を検証する。
#
# step2 の結果（融合 macro-F1・dev 選択）:
#   旧 ASR 仮説で学習した BERT（単体）  0.7537   ← 最良
#   新 ASR 仮説で学習した BERT 4 個     0.7495 / 0.7495 / 0.7516 / 0.7507（平均 0.7504・SD 0.001）
#   新 4 個のアンサンブル               0.7527
#   旧 + 新 4 個のアンサンブル          0.7530
# → 新 4 個のばらつきは SD 0.001。旧との差 0.0034 は 3.4 SD 相当なので偶然ではない。
#   CER 23.3 の汚いテキストで学習した方が、CER 20.7 の綺麗なテキストで評価しても強い。
#
# ただし旧 BERT はいつ・どの引数で学習されたか記録が無く、ハイパラの違いという
# 交絡が残る。そこで:
#   A) 旧仮説で「今のスクリプト・同じ引数」で学習し直す（交絡の排除）
#   B) 旧仮説 + 新仮説 を連結して学習（ノイズ増強。知見が正しければこれが最強）
#
# cron から起動（セッション非依存・oomd 監視外）。BERT 学習は 1 本約 5 分。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
NEWD=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
OLDD=exp/asr_20260713-pureasr/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
STEM=frame_attn-frame2-blk42_same_n5
A="--asr_text $NEWD/eval/text --dev_asr_text $NEWD/train_dev_dec/text"

echo "===== [$(date '+%m-%d %H:%M')] step3 開始 ====="

train() {  # train <保存先> <train_text> [train_text2]
  local out=$1 t1=$2 t2=${3:-}
  [ -s "$out/model.safetensors" ] && { echo "既存: $out"; return 0; }
  local extra=""
  [ -n "$t2" ] && extra="--train_text2 $t2"
  echo "----- [$(date '+%H:%M')] 学習 $out"
  $PY local/turntaking/text_ceiling.py --epochs 2 \
      --train_text "$t1" $extra \
      --dev_text "$NEWD/train_dev_dec/text" --eval_text "$NEWD/eval/text" \
      --save "$out" --out "tt_analysis/$(basename $out).json" \
    && echo "  OK" || echo "  FAIL $out"
}

# A) 旧仮説で再学習（ハイパラ交絡の排除）2 本
train exp/text_bert_oldhyp_a "$OLDD/train_nodup_dec/text"
train exp/text_bert_oldhyp_b "$OLDD/train_nodup_dec/text"
# B) 旧 + 新 連結（ノイズ増強）2 本
train exp/text_bert_mixhyp_a "$OLDD/train_nodup_dec/text" "$NEWD/train_nodup_dec/text"
train exp/text_bert_mixhyp_b "$OLDD/train_nodup_dec/text" "$NEWD/train_nodup_dec/text"

echo "===== [$(date '+%m-%d %H:%M')] 融合で評価 ====="
run() {
  local name=$1 bert=$2 out=tt_analysis/fus_$1.txt
  $PY local/turntaking/frame_fusion.py --stem "$STEM" --bert "$bert" $A > "$out" 2>&1 \
    && grep '融合（dev 選択' "$out" | head -1 | awk -v n="$name" '{printf "  %-18s macroF1 %s\n", n, $(NF-1)}' \
    || echo "  $name 失敗"
}
for x in oldhyp_a oldhyp_b mixhyp_a mixhyp_b; do
  [ -s "exp/text_bert_$x/model.safetensors" ] && run "$x" "exp/text_bert_$x"
done
# 連結学習 2 本のアンサンブル
if [ -s exp/text_bert_mixhyp_a/model.safetensors ] && [ -s exp/text_bert_mixhyp_b/model.safetensors ]; then
  run ens_mixhyp "exp/text_bert_mixhyp_a,exp/text_bert_mixhyp_b"
fi
# 旧再学習 2 本 + 元の旧 BERT のアンサンブル
run ens_oldhyp "exp/text_bert_asr,exp/text_bert_oldhyp_a,exp/text_bert_oldhyp_b"

echo "  （比較）旧 BERT 単体 = 0.7537 / 新 BERT 平均 = 0.7504"
echo "===== [$(date '+%m-%d %H:%M')] step3 完了 ====="
