#!/usr/bin/env bash
# 実験2: 未来VAなし × クラス重みあり [1.92,1.0,1.63]（tight・未来音声なし・マルチタスク）
# これ1本で collect-stats(10) → 学習(11) → デコード(12) → スコアリング(13/result.txt) まで走る。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
export asr_tag=20260625-turntaking-noVA-classweight
export asr_config=myconf/train_asr_turntaking_conformer.yaml
export asr_stats_dir=exp/asr_stats_raw_jp_word_tight
exec ./run_turntaking_tight.sh "$@"
