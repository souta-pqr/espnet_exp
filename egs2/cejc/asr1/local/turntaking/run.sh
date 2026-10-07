#!/usr/bin/env bash
# VAP-lite turn-taking パイプライン: 特徴抽出(凍結ASR) -> ヘッド学習 -> 停止規則評価.
# ASR/エンコーダは一切変更しない(凍結)。conda env は espnet のみ。
set -euo pipefail

PY=/home/kobori/.conda/envs/espnet/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
ASR1=$(cd "$HERE/../.." && pwd)            # egs2/cejc/asr1
cd "$ASR1"

EXP=exp/asr_cif-teaser-methodA-v2-stage1
CONFIG=$EXP/config.yaml
MODEL=$EXP/valid.cer.best.pth
FEAT=$ASR1/local/turntaking/feats         # 特徴シャード保存先
CKPT=$ASR1/local/turntaking/head.pth

# stage 1=抽出, 2=学習, 3=評価。MAX_UTTS>0 でスモークテスト。
STAGE=${1:-1}
STOP=${2:-3}
MAX_UTTS=${MAX_UTTS:-0}
DEVICE=${DEVICE:-cuda}

extract () {  # $1=data名
  local d=$1
  echo "=== 抽出: $d (max_utts=$MAX_UTTS) ==="
  $PY local/turntaking/extract_features.py \
    --config "$CONFIG" --model "$MODEL" \
    --wav_scp data/$d/wav.scp --segments data/$d/segments --tag data/$d/tag \
    --out_dir "$FEAT/$d" --cap 1.0 --horizons 0.2,0.4,0.6,1.0 \
    --max_utts "$MAX_UTTS" --device "$DEVICE"
}

if [ "$STAGE" -le 1 ] && [ "$STOP" -ge 1 ]; then
  extract train_nodup
  extract train_dev
  extract eval
fi

if [ "$STAGE" -le 2 ] && [ "$STOP" -ge 2 ]; then
  echo "=== 学習 ==="
  $PY local/turntaking/train.py \
    --train_dir "$FEAT/train_nodup" --dev_dir "$FEAT/train_dev" \
    --out "$CKPT" --epochs 15 --batch_size 64 --lambda_va 1.0 --device "$DEVICE"
fi

if [ "$STAGE" -le 3 ] && [ "$STOP" -ge 3 ]; then
  echo "=== 評価 (TEASER) ==="
  $PY local/turntaking/evaluate.py --ckpt "$CKPT" --eval_dir "$FEAT/eval" \
    --rule teaser --device "$DEVICE"
  echo "=== 評価 (SPRT) ==="
  $PY local/turntaking/evaluate.py --ckpt "$CKPT" --eval_dir "$FEAT/eval" \
    --rule sprt --device "$DEVICE"
fi
