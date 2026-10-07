#!/usr/bin/env bash
# Phase 4: 新 Stage2（frame2-blk42）の評価。学習完了を待ってから自動で走る。
#
#  1. フレーム単位ダンプ（eval 23,444 / train_dev 14,039）
#  2. 停止規則の評価 → 音声のみ macro-F1（旧 frame2 の 0.7306 と比較）
#  3. 融合 → 音声・テキスト両方を新しくした最終値
#     （旧 1-best 0.7429 / Phase2 テキストのみ更新 0.7500 / 天井 0.7829 と比較）
#
# cron から起動する（user@1609.service の外に置いて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python

D=exp/asr_20260924-turntaking-xfmr-pool_attn-frame2-blk42-same_n5_p30
L=tt_batch_logs/frame2_blk42_s11.log
NEWSTEM=frame_attn-frame2-blk42_same_n5
OLDSTEM=frame_attn-frame2_same_n5
ASR=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave

# ---- 1. 学習完了を待つ（バッチログは追記式なので最後のマーカー以降だけ見る）----
echo "===== [$(date '+%m-%d %H:%M')] Phase4: 学習完了を待機 ====="
while true; do
  c=$(awk '/Phase3-s11/{b=""} {b=b $0 "\n"} END{printf "%s", b}' "$L" 2>/dev/null)
  printf '%s' "$c" | grep -qE "学習 完了" && { echo "学習完了を検知"; break; }
  printf '%s' "$c" | grep -qE "学習 失敗" && { echo "学習が失敗している。中止"; exit 1; }
  pgrep -f "[t]urntaking_asr_train" >/dev/null || { echo "学習プロセス消滅。中止"; exit 1; }
  sleep 300
done

[ -s "$D/valid.loss.ave.pth" ] || { echo "平均モデルが無い。中止"; exit 1; }
echo "最良: $(ls -l $D/valid.loss.best.pth | sed 's/.*-> //')"

# ---- 2. フレームダンプ ----
echo "===== [$(date '+%m-%d %H:%M')] ダンプ開始 ====="
tag_suffix=-frame2-blk42 pool=attn scope=same N=5 max_past=30 \
  sets="eval train_dev" ./run_frame_dump.sh || { echo "ダンプ失敗"; exit 1; }
for ds in eval train_dev; do
  f="tt_preds/${NEWSTEM}_${ds}_valid.loss.ave.tsv"
  [ -s "$f" ] || { echo "ダンプ出力なし: $f"; exit 1; }
done

# ---- 3. 停止規則（音声のみ）: 新旧を並べて出す ----
echo "===== [$(date '+%m-%d %H:%M')] 停止規則の評価 ====="
mkdir -p tt_analysis
$PY local/turntaking/frame_rules.py --step 0.05 \
    --stems "$OLDSTEM" "$NEWSTEM" \
    > tt_analysis/frame_rules_blk42.txt 2>&1 \
  && echo "OK -> tt_analysis/frame_rules_blk42.txt" || echo "停止規則の評価に失敗"

# ---- 4. 融合（音声=新モデル / テキスト=新 Stage1 の認識結果）----
echo "===== [$(date '+%m-%d %H:%M')] 融合 ====="
$PY local/turntaking/frame_fusion.py \
    --stem "$NEWSTEM" \
    --asr_text "$ASR/eval/text" --dev_asr_text "$ASR/train_dev_dec/text" \
    > tt_analysis/frame_fusion_blk42_both.txt 2>&1 \
  && echo "OK -> tt_analysis/frame_fusion_blk42_both.txt" || echo "融合に失敗"

echo "===== [$(date '+%m-%d %H:%M')] Phase4 完了 ====="
