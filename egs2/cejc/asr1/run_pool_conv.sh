#!/usr/bin/env bash
# 【固定率・soft】Compressive Transformer 型の過去圧縮を学習する。
#   出典: J. W. Rae, A. Potapenko, S. M. Jayakumar, C. Hillier, T. P. Lillicrap.
#         "Compressive Transformers for Long-Range Sequence Modelling." ICLR 2020. arXiv:1911.05507
#   機構: 過去 Tp フレームを圧縮率 c ごとのブロックにまとめ ceil(Tp/c) スロットにする（出力 ∝ 過去長）
#   端点: c=1 → frames と完全一致 / c>=Tp → max と数値一致
#
#   pureasr_tag=20260713-pureasr N=1 ./run_pool_conv.sh                # c は平均Tpから自動決定
#   pureasr_tag=20260713-pureasr N=5 ratio=26 ./run_pool_conv.sh       # c を明示
#   pureasr_tag=20260713-pureasr N=1 budget=4 ./run_pool_conv.sh       # 予算 M=4 スロットに合わせる
#
# 圧縮関数（conv=学習あり / mean / max）を変えるには
# myconf/train_asr_turntaking_xfmr_pool_conv.yaml の tt_compress_fn を編集する。
#
# ratio 未指定なら past_bounds から平均過去長を測り c = round(平均Tp / budget) とする。
# 平均 Tp は N で大きく変わる（same: N=1 → 約 44 / N=5 → 約 204 フレーム）ため、
# c を固定したままだと N ごとにスロット数が変わって比較にならない。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export pool=conv
export scope=${scope:-same}
export N=${N:-1}
export max_past=${max_past:-30}
budget=${budget:-8}                 # ratio 未指定時の目標スロット数
train_set=${train_set:-train_nodup}
variant="${scope}_n${N}_p${max_past}"

# 過去入力が学習側に渡る配線があるか（無いと「現発話のみ」のモデルが学習される）
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh || {
    echo "[中止] asr.sh の Stage 11 に past_speech の配線がありません。" >&2
    echo "       この状態で学習すると過去が入らないまま学習されます。" >&2
    exit 1
}

# 圧縮率 c の決定（フレーム = hop 132 サンプル × conv2d 1/4 = 528 サンプル）
if [ -z "${ratio:-}" ]; then
    bounds="dump/raw/${train_set}/past_speech_${variant}_bounds.scp"
    if [ ! -f "${bounds}" ]; then
        echo "[run_pool_conv] past_speech 生成: ${train_set} (${variant})"
        $PY local/build_past_audio.py --seg_dir "data/${train_set}" --dump_dir "dump/raw/${train_set}" \
            --scope "${scope}" --n_past "${N}" --max_past_sec "${max_past}" \
            --out_name "past_speech_${variant}"
    fi
    read -r mean_tp ratio < <(
        awk -v m="${budget}" '{s=0; for(i=2;i<=NF;i++) s+=$i; t+=s; n++}
            END{tp=t/n/528; c=int(tp/m+0.5); if(c<1)c=1; printf "%.0f %d\n", tp, c}' "${bounds}"
    )
    echo "[run_pool_conv] 平均 Tp ≈ ${mean_tp} フレーム → c = round(${mean_tp}/${budget}) = ${ratio}"
fi
export ratio

echo "================================================================"
echo " conv（固定率・soft / Rae+ ICLR2020）"
echo " 圧縮率 c=${ratio}  scope=${scope}  N=${N}  max_past=${max_past}s"
echo " Stage1=${pureasr_tag}  タグ=…-pool_conv_c${ratio}-${variant}"
echo "================================================================"
echo "学習開始後の確認: grep -c past_speech exp/asr_*-pool_conv_c${ratio}-${variant}/config.yaml が 0 なら過去が入っていない"

./run_xfmr_pool.sh "$@"

echo "===== 完了。評価: pool=conv ratio=${ratio} scope=${scope} N=${N} ./eval_xfmr_pool.sh ====="
