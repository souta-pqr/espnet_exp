#!/usr/bin/env bash
# 実験2 を 7ep で打ち切り、blk42 と同条件（valid.loss.ave）で eval をデコードし、
# 続けて実験3（ctc_weight 0.3）を起動する。
#
# 学習を途中で止めると trainer が平均モデルを作らないので、average_nbest_models を
# 手動で呼ぶ。これをやらないと blk42 の CER 21.2（valid.loss.ave 基準）と比較できない。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH

TAG=20260921-pureasr-blk42-sp
E=exp/asr_$TAG
STATS=exp/asr_stats_raw_jp_word_pureasr_sp

echo "===== [$(date '+%m-%d %H:%M')] 実験2 打ち切り処理 開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

# --- 7ep の結果が出るまで待つ（最大 40 分）---
for i in $(seq 1 80); do
    grep -qE "(^|[^0-9])7epoch results" $E/train.log && break
    sleep 30
done
echo "--- 完了エポック ---"
grep -oE "[0-9]+epoch results" $E/train.log | tail -1

# --- 学習を停止 ---
echo "--- [$(date '+%m-%d %H:%M')] 学習を停止 ---"
pkill -f "run_sp.sh" 2>/dev/null
pkill -f "asr_$TAG" 2>/dev/null
sleep 20
pgrep -f "turntaking_asr_train" >/dev/null && { echo "残存プロセスあり、再度停止"; pkill -9 -f "turntaking_asr_train"; sleep 10; }
echo "停止確認: $(pgrep -cf 'run_sp.sh|turntaking_asr_train') プロセス残存"

# --- 平均モデルを手動生成（blk42 と同じ valid.loss.ave を作る）---
echo "--- [$(date '+%m-%d %H:%M')] valid.loss.ave を生成 ---"
python3 - <<'PY'
import torch, yaml
from pathlib import Path
from espnet2.train.reporter import Reporter
from espnet2.main_funcs.average_nbest_models import average_nbest_models

out = Path("exp/asr_20260921-pureasr-blk42-sp")
cfg = yaml.safe_load(open(out / "config.yaml"))
ckpt = torch.load(out / "checkpoint.pth", map_location="cpu", weights_only=False)
rep = Reporter()
rep.load_state_dict(ckpt["reporter"])
print("reporter epochs:", rep.get_epoch())
average_nbest_models(
    output_dir=out,
    reporter=rep,
    best_model_criterion=cfg["best_model_criterion"],
    nbest=cfg["keep_nbest_models"],
)
print("生成物:", sorted(p.name for p in out.glob("*.ave*.pth")))
PY
ls -l $E/valid.loss.ave*.pth 2>&1

# --- eval デコード ---
common=(--asr_stats_dir "$STATS" --ngpu 1
        --speed_perturb_factors "0.9 1.0 1.1"
        --use_streaming false --use_disfluency_detection false
        --use_turntaking_detection true --use_multitask_transducer false
        --use_context_inputs false
        --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word
        --lm_config conf/train_lm.yaml
        --asr_config myconf/train_asr_pureasr_conformer_blk42_sp.yaml
        --asr_tag "$TAG"
        --inference_config myconf/decode_cbs_transducer_bounded.yaml
        --train_set train_nodup --valid_set train_dev --test_sets eval
        --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false
        --inference_asr_model valid.loss.ave.pth)

echo "===== [$(date '+%m-%d %H:%M')] eval デコード ====="
./asr.sh --stage 12 --stop_stage 13 "${common[@]}" \
  && echo "===== [$(date '+%m-%d %H:%M')] デコード 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] デコード 失敗 ====="

echo "--- 対照 blk42 CER 21.2 / WER 28.2 との比較 ---"
grep -A4 "^### CER" $E/RESULTS.md 2>/dev/null | tail -1
grep -A4 "^### WER" $E/RESULTS.md 2>/dev/null | tail -1

# --- 続けて実験3 を起動 ---
echo "===== [$(date '+%m-%d %H:%M')] 実験3 ctc_weight 0.3 を起動 ====="
exec ./tt_batch_logs/run_ctc.sh
