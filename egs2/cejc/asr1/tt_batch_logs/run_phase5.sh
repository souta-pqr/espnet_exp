#!/usr/bin/env bash
# Phase 5: BERT の学習・テスト条件の不整合を解消する。
#
# 背景: 融合のテキスト側 exp/text_bert_asr は「旧 Stage1 の ASR 仮説」で学習されている
#   （text_ceiling.py の --train_text ヘルプ: "ASR 仮説で学習＝条件をそろえる"）。
#   Stage1 を 20260921-pureasr-blk42-sp（CER 23.3→20.7）に替えたことでこの条件が崩れ、
#   BERT は学習時より綺麗なテキストを受け取っている。条件を揃え直して効果を測る。
#
#   Phase 4 時点: 融合 0.7537（BERT は旧 ASR 仮説で学習のまま）／天井 0.7839
#
#  1. train_nodup_dec を新 Stage1 でデコード（197,135 発話・約5時間）
#  2. 新しい仮説で BERT を再学習 → exp/text_bert_asr_blk42sp
#  3. 融合を貼り直して比較
#
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python

TAG=20260921-pureasr-blk42-sp
STATS=exp/asr_stats_raw_jp_word_pureasr_sp
DEC=exp/asr_$TAG/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
BERT=exp/text_bert_asr_blk42sp

echo "===== [$(date '+%m-%d %H:%M')] Phase5 開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 1. train_nodup_dec のデコード ----
if [ -s "$DEC/train_nodup_dec/text" ]; then
  echo "既存: $DEC/train_nodup_dec/text をそのまま使う"
else
  echo "===== [$(date '+%m-%d %H:%M')] train_nodup_dec デコード（約5時間）====="
  cp -n exp/asr_$TAG/RESULTS.md exp/asr_$TAG/RESULTS.md.phase4-bak 2>/dev/null
  ./asr.sh --stage 12 --stop_stage 12 \
      --asr_stats_dir "$STATS" --ngpu 1 \
      --speed_perturb_factors "0.9 1.0 1.1" \
      --use_streaming false --use_disfluency_detection false \
      --use_turntaking_detection true --use_multitask_transducer false \
      --use_context_inputs false \
      --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
      --lm_config conf/train_lm.yaml \
      --asr_config myconf/train_asr_pureasr_conformer_blk42_sp_ext.yaml \
      --asr_tag "$TAG" \
      --inference_config myconf/decode_cbs_transducer_bounded.yaml \
      --train_set train_nodup --valid_set train_dev --test_sets train_nodup_dec \
      --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
      --inference_asr_model valid.loss.ave.pth \
    || { echo "デコード失敗"; exit 1; }
  [ -s "$DEC/train_nodup_dec/text" ] || { echo "デコード出力なし"; exit 1; }
fi
echo "train_nodup_dec: $(wc -l < $DEC/train_nodup_dec/text) 発話"

# ---- 2. BERT 再学習（新しい ASR 仮説で条件をそろえる）----
echo "===== [$(date '+%m-%d %H:%M')] BERT 再学習 ====="
$PY local/turntaking/text_ceiling.py --epochs 2 \
    --train_text "$DEC/train_nodup_dec/text" \
    --dev_text   "$DEC/train_dev_dec/text" \
    --eval_text  "$DEC/eval/text" \
    --save "$BERT" \
    --out tt_analysis/text_bert_blk42sp.json \
  || { echo "BERT 再学習に失敗"; exit 1; }
[ -s "$BERT/model.safetensors" ] || { echo "BERT の重みが無い"; exit 1; }

# ---- 3. 融合を貼り直す ----
echo "===== [$(date '+%m-%d %H:%M')] 融合（BERT 条件そろえ）====="
$PY local/turntaking/frame_fusion.py \
    --stem frame_attn-frame2-blk42_same_n5 \
    --bert "$BERT" \
    --asr_text "$DEC/eval/text" --dev_asr_text "$DEC/train_dev_dec/text" \
    > tt_analysis/frame_fusion_blk42_bertmatch.txt 2>&1 \
  && echo "OK -> tt_analysis/frame_fusion_blk42_bertmatch.txt" || echo "融合に失敗"

echo "===== [$(date '+%m-%d %H:%M')] Phase5 完了 ====="
