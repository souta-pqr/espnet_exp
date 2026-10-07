#!/usr/bin/env bash
# run_audio_side.sh が C1（Stage1 blk60 の学習とデコード）を終えて C2 に入ったら止める（2026-10-05）。
# 分割を作り直すことにしたため、古い分割での C2 は回さない。C1 の CER は block_size を決める材料として残す。
# cron から一度だけ起動する。
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
L=tt_batch_logs/audio_side.log
echo "===== [$(date '+%m-%d %H:%M')] C2 の手前で止める見張り 開始 ====="
while true; do
  if [ -e tt_batch_logs/audio_side_C1_decode.done ] && grep -q "C2: Stage2 frame2 blk60 学習" "$L"; then
    pkill -f "bash /home/kobori/2025/espnet/egs2/cejc/asr1/tt_batch_logs/run_audio_sid[e].sh"
    sleep 2
    pkill -f "asr\.s[h] --stage 11 .*frame2_blk60"
    pkill -f "run\.p[l] --name exp/asr_20260930-turntaking-xfmr-pool_attn-frame2-blk60"
    pkill -f "turntaking_asr_trai[n].*frame2-blk60"
    pkill -f "bin\.launc[h].*frame2-blk60"
    echo "===== [$(date '+%m-%d %H:%M')] C2 を止めた ====="
    exit 0
  fi
  pgrep -f "bash /home/kobori/2025/espnet/egs2/cejc/asr1/tt_batch_logs/run_audio_sid[e].sh" >/dev/null \
    || { echo "===== [$(date '+%m-%d %H:%M')] ドライバが先に終わった ====="; exit 0; }
  sleep 60
done
