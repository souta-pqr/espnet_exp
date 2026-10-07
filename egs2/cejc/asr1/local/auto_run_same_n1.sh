#!/usr/bin/env bash
# GPU/学習が空くのを待ってから N=1（同一話者・過去文脈）を自動実行するウォッチャ。
# 走行中の ctx 学習・抽出・デコードが全て終わり、3分連続でアイドルになったら起動する。
# （ctx_vec.scp は方式間で共有されるため、他ジョブと同時起動を避ける目的）
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)/..
cd "$HERE"
LOG="$HERE/local/auto_run_same_n1.log"

busy() {
    pgrep -af 'turntaking_asr_train|extract_context_vectors|bin\.asr_inference|run_ctxvec\.sh|run_exp_pastctx' \
        | grep -v 'auto_run_same_n1' | grep -v grep >/dev/null
}

echo "[auto-n1] 監視開始 $(date)" | tee -a "$LOG"
idle=0
while true; do
    if busy; then
        idle=0
    else
        idle=$((idle+1))
        echo "[auto-n1] idle ${idle}/3 $(date)" >>"$LOG"
        [ "$idle" -ge 3 ] && break
    fi
    sleep 60
done

echo "[auto-n1] アイドル確認 → N=1 開始 $(date)" | tee -a "$LOG"
./run_exp_pastctx_same_n1.sh >>"$LOG" 2>&1
echo "[auto-n1] N=1 完了 $(date)" | tee -a "$LOG"
