#!/usr/bin/env bash
# アブレーション（2026-10-26）：gm（20261021-...-gm）の設定から 1 か所だけ変えて Stage2 を学習し，フレームダンプを作る。
#   abnp：過去の文脈なし（tt_past_pool: none）
#   abnt：区間末の後の位置で学習しない（tt_tail_readout: false．入力は gm と同じ）
#   abna：ASR の事前学習なし（init_param なし・エンコーダも学習．tt_adapt: full）
#   abnl：abna の学習率を Stage1 の ASR と同じにしたもの（lr 0.0005・warmup 5000．2026-10-27）
# 学習の手順（データ・ラベル・クラス重み・seed・lr・patience・max_epoch）は gm と同じ。
# 評価（eot_eval・2 段の規則・bootstrap）は silver11 で gm と同じ手順で行う。
# 各段の完了時に tt_batch_logs/ab_<名前>_<段>.done を置く。
# 起動：tmux の外から setsid nohup bash tt_batch_logs/run_ab.sh <名前> >> tt_batch_logs/ab_<名前>.log 2>&1 &
set -u
AB=$1
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
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
TAG=20261026-turntaking-xfmr-pool_attn-frame2-blk42xh2-${AB}-same_n5_p30
STATS=exp/asr_stats_raw_jp_g_word_pool_attn_same_n5_p30_tail
TR=train_gx
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
         --lm_train_text "data/$TR/text" --use_lm false --use_word_lm false
         --inference_asr_model valid.loss.ave.pth
         --asr_config "$CFG" --asr_tag "$TAG")

if ! done_ train; then
  log "Stage2 学習（$TAG）"
  ./asr.sh --stage 11 --stop_stage 11 "${tt_args[@]}" \
    && [ -s "exp/asr_$TAG/valid.loss.ave.pth" ] || { echo "Stage2 学習に失敗"; exit 1; }
  mark train
fi

if ! done_ dump; then
  log "フレームダンプ（eval_g / dev_g，区間の長さ + 5 秒）"
  tag_suffix=-frame2-blk42xh2-$AB post=5.0 sets="eval_g dev_g" ./run_frame_dump.sh
  for ds in eval_g dev_g; do
    [ -s "${P}-${AB}_same_n5_${ds}_valid.loss.ave_post5.0.tsv" ] || { echo "ダンプ出力なし: $ds"; exit 1; }
  done
  mark dump
fi
log "AB $AB 完了"
