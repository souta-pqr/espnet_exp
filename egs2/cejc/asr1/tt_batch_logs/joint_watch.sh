#!/usr/bin/env bash
# 共同学習の記録係。5 エポック刻みの要約と異常を tt_analysis/joint_watch.log に追記する。
# ハーネスから切り離して動かすので、セッションの都合では止まらない。
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
J=exp/asr_20260916-turntaking-xfmr-pool_attn-joint-same_n5_p30/train.log
B=exp/asr_20260913-turntaking-xfmr-pool_attn-frame2-same_n5_p30/train.log
OUT=tt_analysis/joint_watch.log
PID=4140913
mkdir -p tt_analysis

val() { echo "$1" | sed "s/.*\[$2\]//" | grep -oE "$3=[0-9.e+-]+" | head -1 | cut -d= -f2; }

last=0
[ -f "$OUT" ] && last=$(grep -oE "^[0-9]+ep" "$OUT" | tail -1 | tr -d 'ep')
: "${last:=0}"

while true; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[$(date '+%m-%d %H:%M')] ジョブ終了: PID $PID が消えました" >> "$OUT"
        tail -5 tt_batch_logs/joint_train.log >> "$OUT"
        exit 0
    fi
    if [ -f "$J" ]; then
        cur=$(grep -oE "[0-9]+epoch results" "$J" | tail -1 | grep -oE "^[0-9]+")
        if [ -n "${cur:-}" ]; then
            while [ $((last + 5)) -le "$cur" ]; do
                e=$((last + 5))
                lj=$(grep -E "(^|[^0-9])${e}epoch results" "$J" | tail -1)
                lb=$(grep -E "(^|[^0-9])${e}epoch results" "$B" | tail -1)
                printf '%sep [%s] joint tr_acc=%s va_loss=%s va_acc=%s | frame2 va_loss=%s va_acc=%s\n' \
                    "$e" "$(date '+%m-%d %H:%M')" \
                    "$(val "$lj" train tag_acc)" "$(val "$lj" valid loss_tag)" "$(val "$lj" valid tag_acc)" \
                    "$(val "$lb" valid loss_tag)" "$(val "$lb" valid tag_acc)" >> "$OUT"
                last=$e
            done
        fi
        err=$(grep -nE "Traceback|RuntimeError|out of memory|CUDA error|AssertionError" "$J" | tail -3)
        if [ -n "$err" ]; then
            echo "[$(date '+%m-%d %H:%M')] 異常検出:" >> "$OUT"; echo "$err" >> "$OUT"; exit 0
        fi
    fi
    sleep 120
done
