#!/usr/bin/env bash
# Stage1 ASR 延長学習の記録係。エポックごとに valid loss と最良更新を追記する。
# 純粋 ASR は loss_transducer を出す（loss_tag ではない）。
set -u
cd /home/kobori/2025/espnet/egs2/cejc/asr1
J=exp/asr_20260918-pureasr-ext/train.log
OUT=tt_analysis/asr_ext_watch.log
PID=2594821        # 09-20 05:56 再開（SSH セッションスコープ内。user@.service の外なので oomd 対象外）
BEST=13.269        # 旧モデル 20260713-pureasr の 47 エポック（延長前の到達点）
mkdir -p tt_analysis

last=0
[ -f "$OUT" ] && last=$(grep -oE "^[0-9]+ep" "$OUT" | tail -1 | tr -d 'ep')
: "${last:=50}"

while true; do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[$(date '+%m-%d %H:%M')] ジョブ終了: PID $PID が消えました" >> "$OUT"
        tail -6 tt_batch_logs/asr_ext.log >> "$OUT"
        exit 0
    fi
    cur=$(grep -oE "[0-9]+epoch results" "$J" 2>/dev/null | tail -1 | grep -oE "^[0-9]+")
    if [ -n "${cur:-}" ]; then
        while [ $((last + 1)) -le "$cur" ]; do
            e=$((last + 1))
            l=$(grep -E "(^|[^0-9])${e}epoch results" "$J" | tail -1)
            if [ -n "$l" ]; then
                vl=$(echo "$l" | sed 's/.*\[valid\]//' | grep -oE 'loss=[0-9.]+' | head -1 | cut -d= -f2)
                upd=$(grep -A 2 -E "(^|[^0-9])${e}epoch results" "$J" | grep -c "best model has been updated")
                mark=""
                [ "${upd:-0}" -gt 0 ] && mark="  ← 最良更新"
                awk -v a="$vl" -v b="$BEST" 'BEGIN{if(a!=""&&a+0<b+0) print ""}' >/dev/null
                better=$(awk -v a="${vl:-99}" -v b="$BEST" 'BEGIN{print (a+0<b+0)?"○":" "}')
                printf '%sep [%s] valid loss=%s 旧最良比 %s%s\n' \
                    "$e" "$(date '+%m-%d %H:%M')" "${vl:-?}" "$better" "$mark" >> "$OUT"
            fi
            last=$e
        done
    fi
    if grep -qE "Traceback|RuntimeError|out of memory|CUDA error" "$J" 2>/dev/null; then
        echo "[$(date '+%m-%d %H:%M')] 異常検出" >> "$OUT"
        grep -nE "Traceback|RuntimeError|out of memory|CUDA error" "$J" | tail -3 >> "$OUT"
        exit 0
    fi
    sleep 300
done
