#!/usr/bin/env bash
# run_audio_side.sh（A → B → B40 → C1 → C2）の完了を待ち、C2 で blk60 が効いていれば
# D（blk60 ＋ Transformer 2 層ヘッド）を学習・評価する。2026-10-05 追加。
#
# 判定: C2（blk60・selfattn）が、同じ selfattn の blk42（frame2-blk42）を
#       融合 0.7537 か 音声のみ 0.7354 のどちらかで上回れば D を回す。
# 最後に、早期確定の規則を dev で選び直す（commit_rules.py、持続 3 に限定）。
#
# cron から起動（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kobori/.conda/envs/espnet/bin/python
M=tt_batch_logs/audio_side
ASR60=exp/asr_20260930-pureasr-blk60-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
TT_STATS=exp/asr_stats_raw_jp_word_pool_attn_same_n5_p30_tail
BASE_STEM=frame_attn-frame2-blk42_same_n5
D_TAG=20260930-turntaking-xfmr-pool_attn-frame2-blk60xh2-same_n5_p30
D_CFG=myconf/.gen_pool_attn-frame2-blk60xh2_same_n5_p30.yaml
D_STEM=frame_attn-frame2-blk60xh2_same_n5

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
echo; log "AUDIO_SIDE_D 開始"
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- 1. 本体の完了を待つ（完了印と、本体プロセスの生存で判断）----
while [ ! -e "${M}_C2_eval.done" ]; do
  if ! pgrep -f "bash $HERE/tt_batch_logs/run_audio_side.s[h]" >/dev/null; then
    echo "本体が C2 評価の前に止まっている。中止"; exit 1
  fi
  sleep 600
done
log "C2 評価の完了を検知"

# ---- 2. 判定 ----
read -r a60 f60 < <(awk '/区間検出まで待つ/ && /音声のみ/{a=$(NF-1)} /区間検出まで待つ/ && /融合（dev 選択/{f=$(NF-1)} END{print a, f}' \
                    tt_analysis/frame_fusion_C2.txt)
echo "C2: 音声のみ $a60 / 融合 $f60 （blk42 selfattn: 0.7354 / 0.7537）"
if awk -v a="$a60" -v f="$f60" 'BEGIN{exit !(a > 0.7354 || f > 0.7537)}'; then
  echo "blk60 は効いた → D を回す"
else
  echo "blk60 は効かなかった → D は回さない。最終構成は B（blk42 ＋ Transformer 2 層）"
  log "AUDIO_SIDE_D 完了（D なし）"; exit 0
fi

# ---- 3. D の学習（OOM なら batch_bins 半分・accum_grad 倍で最大 2 回）----
if [ ! -e "${M}_D_train.done" ]; then
  log "D: blk60 ＋ Transformer 2 層ヘッド 学習"
  for try in 0 1 2; do
    ./asr.sh --stage 11 --stop_stage 11 \
        --asr_stats_dir "$TT_STATS" --ngpu 1 \
        --use_streaming false --use_disfluency_detection false \
        --use_turntaking_detection true --use_multitask_transducer false \
        --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
        --lm_config conf/train_lm.yaml \
        --inference_config myconf/decode_cbs_transducer_bounded.yaml \
        --train_set train_nodup --valid_set train_dev --test_sets eval \
        --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false \
        --inference_asr_model valid.loss.ave.pth \
        --asr_config "$D_CFG" --asr_tag "$D_TAG" \
      && [ -s "exp/asr_$D_TAG/valid.loss.ave.pth" ] && { touch "${M}_D_train.done"; break; }
    if tail -n 300 "exp/asr_$D_TAG/train.log" 2>/dev/null | grep -q "CUDA out of memory" && [ "$try" != 2 ]; then
      bb=$(awk '/^batch_bins:/{print $2}' "$D_CFG"); ag=$(awk '/^accum_grad:/{print $2}' "$D_CFG")
      echo "CUDA OOM → batch_bins $bb→$((bb / 2)) / accum_grad $ag→$((ag * 2)) でやり直す"
      sed -i -E "s/^batch_bins: $bb/batch_bins: $((bb / 2))/; s/^accum_grad: $ag/accum_grad: $((ag * 2))/" "$D_CFG"
      mv "exp/asr_$D_TAG" "exp/asr_$D_TAG.oom-bak$try"
    else
      echo "D の学習に失敗"; exit 1
    fi
  done
fi

# ---- 4. D の評価 ----
if [ ! -e "${M}_D_eval.done" ]; then
  log "D: フレームダンプ"
  tag_suffix=-frame2-blk60xh2 pool=attn scope=same N=5 max_past=30 sets="eval train_dev" ./run_frame_dump.sh
  for ds in eval train_dev; do
    [ -s "tt_preds/${D_STEM}_${ds}_valid.loss.ave.tsv" ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  log "D: 停止規則"
  $PY local/turntaking/frame_rules.py --step 0.05 --stems "$BASE_STEM" "$D_STEM" \
      > tt_analysis/frame_rules_D.txt 2>&1 || { echo "停止規則に失敗"; exit 1; }
  log "D: 融合"
  $PY local/turntaking/frame_fusion.py --stem "$D_STEM" --bert exp/text_bert_asr \
      --asr_text "$ASR60/eval/text" --dev_asr_text "$ASR60/train_dev_dec/text" \
      > tt_analysis/frame_fusion_D.txt 2>&1 || { echo "融合に失敗"; exit 1; }
  grep -E "音声のみ|融合（dev 選択" tt_analysis/frame_fusion_D.txt | grep "区間検出まで待つ"
  touch "${M}_D_eval.done"
fi

# ---- 5. 早期確定の規則（dev 選択・持続 3）----
log "早期確定の規則（C2 / D）"
$PY local/turntaking/commit_rules.py --persist 3 --asr_dir "$ASR60" \
    --stems frame_attn-frame2-blk60_same_n5 "$D_STEM" \
    > tt_analysis/commit_rules_C2_D_persist3.txt 2>&1 \
  && echo "OK -> tt_analysis/commit_rules_C2_D_persist3.txt" || echo "早期確定の規則に失敗"

log "AUDIO_SIDE_D 完了"
