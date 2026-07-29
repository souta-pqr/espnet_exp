#!/usr/bin/env bash
# session(複数話者) N=4 seqcat/current を epoch42 から50まで再開（元の日付タグ 20260711 を固定）。
# past_speech は sox不使用の concat_sound 形式に更新済み＝DataLoaderデッドロックしない。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export asr_tag=20260711-turntaking-noVA-classweight-ctx_session_n4_p30_seqcat_current
echo "[resume] tag=${asr_tag}"
echo "[resume] 再開元: $(readlink exp/${asr_tag}/latest.pth 2>/dev/null || echo '?')"
pool_range=current NS=4 max_past=30 ./run_exp_pastctx_session_seqcat_series.sh --stage 11 "$@"
