#!/usr/bin/env bash
# 【新提案】Transducer デコーダの状態をタグヘッドに接続する。
#   根拠: 発話区間末タグは LLM が「書き起こし＋文脈」から付けた意味・語用の判断であり、
#         韻律はラベル生成過程に入っていない。判断に必要な言語情報を担うのはデコーダだが、
#         現行モデルの head は encoder 出力 C'_T しか見ておらず、デコーダ状態を使っていない。
#         ASR で既に計算済みなので追加コストはほぼゼロ。
#   実装: model_conf の tt_use_dec: true。デコーダは凍結のまま状態列を線形射影し、
#         [過去 ; デコーダ状態 ; 現発話フレーム] として head の self-attention に入れる。
#   過去要約は pool= で選ぶ（既定 attn ＝ N=5 で最良だった凍結手法）。
#
#   pureasr_tag=20260713-pureasr N=5 ./run_pool_dec.sh
#   pureasr_tag=20260713-pureasr N=0 ./run_pool_dec.sh   # 過去なしで純粋にデコーダ効果を見る
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    exit 1
}
grep -q "tt_use_dec" ../../../espnet2/asr/espnet_model_turntaking_xfmr.py || {
    echo "[中止] espnet_model_turntaking_xfmr.py に tt_use_dec がありません。" >&2
    exit 1
}
export pool=${pool:-attn}
export base_cfg=myconf/train_asr_turntaking_xfmr_pool_${pool}_dec.yaml
[ -f "$base_cfg" ] || { echo "[中止] ${base_cfg} がありません" >&2; exit 1; }
export tag_suffix=-dec
echo "=== デコーダ状態を head へ接続  過去要約=${pool}  scope=${scope:-same} N=${N:-5}"
exec ./run_xfmr_pool.sh "$@"
