#!/usr/bin/env bash
# N-best 融合を新 Stage1（20260921-pureasr-blk42-sp）で取り直す。
#
# 旧 Stage1 では 1-best 0.7429 → N-best 0.7461（+0.0032）。融合側で唯一効いた改良。
# 新 Stage1 の 1-best 融合は 0.7537（dev 選択・音声 0.55）。
#
#  1. eval と train_dev_dec を N=5 でデコード
#  2. ジョブ別の 2〜5 位を連結（asr.sh は 1best_recog しか集約しない）
#  3. nbest_fusion.py
#
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python

TAG=20260921-pureasr-blk42-sp
STATS=exp/asr_stats_raw_jp_word_pureasr_sp
DEC1=exp/asr_$TAG/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
DECN=exp/asr_$TAG/decode_cbs_transducer_bounded_nbest5_asr_model_valid.loss.ave

echo "===== [$(date '+%m-%d %H:%M')] NBEST_BLK42 開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 1. N=5 デコード ----
cp -n exp/asr_$TAG/RESULTS.md exp/asr_$TAG/RESULTS.md.nbest-bak 2>/dev/null
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
    --inference_config myconf/decode_cbs_transducer_bounded_nbest5.yaml \
    --train_set train_nodup --valid_set train_dev --test_sets "eval train_dev_dec" \
    --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth \
  || { echo "デコード失敗"; exit 1; }

# ---- 2. 2〜5 位を連結 ----
for s in eval train_dev_dec; do
  echo "--- $s"
  ./local/turntaking/merge_nbest.sh "$DECN" "$s" 5 || { echo "連結失敗 $s"; exit 1; }
done

# ---- 3. 融合 ----
echo "===== [$(date '+%m-%d %H:%M')] N-best 融合 ====="
$PY local/turntaking/nbest_fusion.py \
    --stem frame_attn-frame2-blk42_same_n5 --bert exp/text_bert_asr \
    --decode_dir "$DECN" \
    --onebest "$DEC1/eval/text" --dev_onebest "$DEC1/train_dev_dec/text" \
    --w 0.55 \
    > tt_analysis/nbest_fusion_blk42.txt 2>&1 \
  && echo "OK -> tt_analysis/nbest_fusion_blk42.txt" || { echo "融合に失敗"; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] NBEST_BLK42 完了 ====="
