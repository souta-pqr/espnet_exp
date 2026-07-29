#!/usr/bin/env bash
# 実験3: 過去文脈（同一話者の直前N発話の特徴ベクトルを区間末ヘッドに注入・ASR不変・未来なし）
# N=1 と N=2 を順に学習。各 N で ctx_vec抽出 → 学習 → デコード → スコアリング(result.txt) まで走る。
# クラス重みは既定で「あり」（実験2と揃える）。重みなしで見たい場合は weight=none をつける。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"

for n in 1 2; do
    echo "===== 実験3: 過去文脈 N=${n} ====="
    N="${n}" ./run_ctxvec.sh "$@"
done
