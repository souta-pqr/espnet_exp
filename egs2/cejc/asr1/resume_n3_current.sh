#!/usr/bin/env bash
# N=3 seqcat/current を epoch45 から 50 まで再開（元の日付タグ 20260710 を固定）。
# 日付タグ($(date))のせいで別dirになり最初から始まる問題を回避する。
# ハングした場合（GPU 0% のまま20分以上停止）は Ctrl-C で止めて本スクリプトを再実行すれば、
# 最後に保存された epoch（checkpoint.pth）から続きます。50まで進めば自動でデコード→CER。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export asr_tag=20260710-turntaking-noVA-classweight-ctx_same_n3_p30_seqcat_current
echo "[resume] tag=${asr_tag}"
echo "[resume] 再開元: $(readlink exp/${asr_tag}/latest.pth 2>/dev/null || echo '?')"
pool_range=current NS=3 max_past=30 ./run_exp_pastctx_same_seqcat_series.sh --stage 11 "$@"
