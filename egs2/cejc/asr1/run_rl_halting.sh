#!/usr/bin/env bash
# RL halting policy（学習型停止方策・系統②）の実験ランナー。
# 学習済みの発話区間末モデル（cls ヘッド）の p_t の上に、停止方策(Controller)を REINFORCE で学習し評価する。
# フル ASR 再学習は不要。詳細: turntaking_rl_halting_method.md
#
# 使い方:
#   ./run_rl_halting.sh
#   asr_tag=20260615-turntaking-multitask-classweight ./run_rl_halting.sh
#   lambdas="0.2 0.5 1.0 2.0" epochs=30 ./run_rl_halting.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}

# どの学習済みモデルを使うか
asr_tag=${asr_tag:-20260615-turntaking-multitask-classweight}
ckpt=${ckpt:-valid.tag_acc.best.pth}
E="exp/asr_${asr_tag}"

# 実験条件
lambdas=${lambdas:-"0.1 0.3 0.5 1.0"}   # 遅延ペナルティ λ の sweep
cost=${cost:-1.0}                        # 誤判定ペナルティ c
epochs=${epochs:-20}
train_dir=${train_dir:-dump/raw/train_dev_trail}
eval_dir=${eval_dir:-dump/raw/eval_trail}

[ -f "$E/config.yaml" ] || { echo "config が無い: $E/config.yaml" >&2; exit 1; }
[ -f "$E/$ckpt" ]       || { echo "ckpt が無い: $E/$ckpt" >&2; exit 1; }

echo "=== RL halting: model=${asr_tag}/${ckpt}  λ in [${lambdas}]  (c=${cost}, epochs=${epochs}) ==="
for L in ${lambdas}; do
  echo "----- λ=${L} -----"
  $PY local/turntaking/rl_halting.py \
      --config "$E/config.yaml" --model "$E/$ckpt" \
      --train_dir "$train_dir" --eval_dir "$eval_dir" \
      --lambda_delay "$L" --cost "$cost" --epochs "$epochs" "$@"
done
