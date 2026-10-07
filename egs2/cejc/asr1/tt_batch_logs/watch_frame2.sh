#!/usr/bin/env bash
# frame2-blk42 学習の終端待機。完了・失敗・プロセス消滅のいずれかで 1 行だけ書いて終わる。
# cron から起動して私のセッションから独立させる（背景タスク終了で連れて行かれないように）。
# 注意: バッチログは追記式なので、必ず最後の Phase3-s11 マーカー以降だけを見る。
#       （これを怠って過去の失敗を拾う誤報を 2 回出した）
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
D=exp/asr_20260924-turntaking-xfmr-pool_attn-frame2-blk42-same_n5_p30
L=tt_batch_logs/frame2_blk42_s11.log
OUT=tt_batch_logs/frame2_watch_result.txt
: > "$OUT"
while true; do
  c=$(awk '/Phase3-s11/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$L" 2>/dev/null)
  if printf '%s' "$c" | grep -qE "学習 完了"; then r="COMPLETE 学習完了"; break; fi
  if printf '%s' "$c" | grep -qE "学習 失敗"; then r="FAIL ジョブ失敗"; break; fi
  if ! pgrep -f "[t]urntaking_asr_train" >/dev/null; then r="FAIL 学習プロセス消滅"; break; fi
  sleep 300
done
{
  echo "[$(date '+%m-%d %H:%M')] $r"
  echo "到達エポック: $(grep -oE '[0-9]+epoch results' "$D/train.log" 2>/dev/null | tail -1)"
  grep -oE "[0-9]+epoch results.*valid\] loss_tag=[0-9.]+" "$D/train.log" 2>/dev/null \
    | sed 's/ results.*valid\]/ /' | tail -5
} >> "$OUT"
