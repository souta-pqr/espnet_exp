#!/usr/bin/env bash
# Phase 6 step2: BERT 個別 4 種・新アンサンブル・旧込みアンサンブルを融合で比較。
# 目的 (a) Phase5 の −0.0041 がばらつきか本物かの切り分け
#      (b) 確率平均で 0.7537 を超えられるか
# シード学習の完了を待ってから走る。cron から起動（セッション非依存・oomd 監視外）。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
DEC=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
STEM=frame_attn-frame2-blk42_same_n5
A="--asr_text $DEC/eval/text --dev_asr_text $DEC/train_dev_dec/text"

echo "===== [$(date '+%m-%d %H:%M')] step2: シード学習の完了を待機 ====="
for i in $(seq 1 240); do
  grep -q "完了 =====" tt_batch_logs/bert_seeds.log 2>/dev/null && break
  pgrep -f "[r]un_bert_seeds" >/dev/null || { echo "シード学習が消滅"; break; }
  sleep 30
done
NEW=exp/text_bert_asr_blk42sp
for s in 2 3 4; do [ -s "exp/text_bert_asr_blk42sp_s$s/model.safetensors" ] && NEW="$NEW,exp/text_bert_asr_blk42sp_s$s"; done
echo "揃った新 BERT: $NEW"

run() {  # run <名前> <bertの指定>
  local name=$1 bert=$2 out=tt_analysis/fus_$1.txt
  echo "----- [$(date '+%H:%M')] $name"
  $PY local/turntaking/frame_fusion.py --stem "$STEM" --bert "$bert" $A > "$out" 2>&1 \
    && grep -E "融合（dev 選択" "$out" | head -1 | awk -v n="$name" '{printf "  %-22s macroF1 %s\n", n, $(NF-1)}' \
    || echo "  $name 失敗"
}

# 1) 個別（ばらつきの幅を見る）
run bert_new1 exp/text_bert_asr_blk42sp
for s in 2 3 4; do
  [ -s "exp/text_bert_asr_blk42sp_s$s/model.safetensors" ] && run "bert_new$s" "exp/text_bert_asr_blk42sp_s$s"
done
# 2) 新アンサンブル
run ens_new "$NEW"
# 3) 旧込みアンサンブル（学習テキストの誤りの質が違う＝多様性）
run ens_mix "exp/text_bert_asr,$NEW"
# 4) 旧単体（Phase4 の 0.7537 の再現確認）
run bert_old exp/text_bert_asr

echo "===== [$(date '+%m-%d %H:%M')] step2 完了 ====="
