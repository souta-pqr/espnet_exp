# silver13 用（2026-11-07）：silver12 の run_gxm.sh の複製．asr.sh の代わりに asr_gxm.sh（silver12 の asr.sh の複製．SPEECH_TYPE を読む）を使う
#!/usr/bin/env bash
# gtm（2026-11-04，silver12）：gt から続けて，間を含む長い発話を足したデータ train_gxm（make_gxm.py）で 4 エポック学習し，フレームダンプを作る。
# 音声は「前の区間 間 今の区間」の 3 つの flac を concat_sound で時間方向につないで読む（SPEECH_TYPE）。
# 起動：tmux の外から setsid nohup bash tt_batch_logs/run_gxm.sh gtm >> tt_batch_logs/ab_gtm.log 2>&1 < /dev/null &
set -u
AB=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)   # silver11・13 は ~/2025/espnet，silver12 は ~/2026/espnet_tt の下
cd "$HERE"
if [ -x /home/kobori/.conda/envs/espnet/bin/python ]; then
  ENV=/home/kobori/.conda/envs/espnet/bin               # silver11
else
  ENV=/home/kobori/anaconda3/envs/espnet/bin            # silver13（espnet は ~/2025/espnet を PYTHONPATH で使う）
  export PYTHONPATH=/home/kobori/2025/espnet
fi
export PATH=$ENV:$PATH
export PY=$ENV/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=tt_batch_logs/ab_$AB
CFG=myconf/.gen_pool_attn-frame2-blk42xh2-${AB}_same_n5_p30.yaml
TAG=20261104-turntaking-xfmr-pool_attn-frame2-blk42xh2-${AB}-same_n5_p30
STATS=exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail_gxm
TR=train_gxm
export SPEECH_TYPE=concat_sound
P=tt_preds/frame_attn-frame2-blk42xh2

log() { echo "===== [$(date '+%m-%d %H:%M')] $* ====="; }
done_() { [ -e "${M}_$1.done" ]; }
mark() { touch "${M}_$1.done"; }
[ -e "$CFG" ] || { echo "設定がない: $CFG"; exit 1; }
echo; log "AB $AB 開始（$(hostname)）"
echo "cgroup: $(cat /proc/self/cgroup)"

tt_args=(--asr_stats_dir "$STATS" --ngpu 1
         --use_streaming false --use_disfluency_detection false
         --use_turntaking_detection true --use_multitask_transducer false
         --nj 16 --inference_nj 16 --lang jp_g --feats_type raw --token_type word
         --lm_config conf/train_lm.yaml
         --inference_config myconf/decode_cbs_transducer_bounded.yaml
         --train_set "$TR" --valid_set dev_g --test_sets eval_g
         --lm_train_text "data/train_gx/text" --use_lm false --use_word_lm false
         --inference_asr_model valid.loss.ave.pth
         --asr_config "$CFG" --asr_tag "$TAG")

if ! done_ train; then
  NEED_MB=${NEED_MB:-14000}
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$NEED_MB" ]; do sleep 600; done
  init=$(sed -n '/^init_param:/{n;s/^ *- *\([^ ]*\).*/\1/p}' "$CFG")
  [ -e "$init" ] || { echo "初期値がない: $init"; exit 1; }
  log "Stage2 学習（$TAG）"
  ./asr_gxm.sh --stage 11 --stop_stage 11 "${tt_args[@]}" \
    && [ -s "exp/asr_$TAG/valid.loss.ave.pth" ] || { echo "Stage2 学習に失敗"; exit 1; }
  mark train
fi

if ! done_ dump; then
  log "フレームダンプ（eval_g / dev_g，区間の長さ + 5 秒）"
  tag_suffix=-frame2-blk42xh2-$AB post=5.0 sets="eval_g dev_g" ./run_frame_dump.sh
  for ds in eval_g dev_g; do
    [ $(wc -l < "${P}-${AB}_same_n5_${ds}_valid.loss.ave_post5.0.tsv" 2>/dev/null || echo 0) -gt 1000 ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  mark dump
fi
log "AB $AB 完了"
