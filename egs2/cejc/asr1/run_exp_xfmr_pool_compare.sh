#!/usr/bin/env bash
# 「どの過去要約が有効か」を **スロット予算を揃えて** 比較する（過去発話数 N 固定）。
#
#   予算 M スロットの下で以下を同一条件で学習する:
#     attn   固定長・soft  M=1 に潰す既存手法（下限の参照点）
#     frames 圧縮なし      Np = Tp（上限の参照点。c=1 に相当）
#     conv   固定率・soft  圧縮率 c = round(平均Tp / M) → 平均 Np ≈ M   [Rae+ ICLR2020  arXiv:1911.05507]
#     query  固定長・soft  Np = M                                        [Lee+ ICML2019 arXiv:1810.00825 /
#                                                                         Jaegle+ ICML2021 arXiv:2103.03206]
#     topk   固定長・hard  Np = min(M, Tp)                               [Rae+ ICLR2020 "most-used"]
#     utt    会話単位      Np = 有効発話数（k=1）。直接の出典はないが RQ4（区切りは会話単位か）の対照
#
#   budget=8 N=1 scope=same max_past=30 pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_compare.sh
#   POOLS="conv query topk" budget=8 N=5 scope=same pureasr_tag=... ./run_exp_xfmr_pool_compare.sh
#   force=1 ...  # 既に学習済みのタグも無視して再学習する
#
# conv の圧縮率は past_bounds から測った **平均過去長** をもとに決める。
# 平均 Tp は N と scope で大きく変わるため（same: N=1 → 約 44 フレーム / N=5 → 約 204 フレーム）、
# c を固定すると手法間でスロット数が揃わず「どれが有効か」の比較にならない。
#
# つまみを振る掃引は run_exp_xfmr_pool_compress_sweep.sh、N を振る掃引は run_exp_xfmr_pool_series.sh。
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
export pureasr_tag=${pureasr_tag:?Stage1のタグを指定 例) pureasr_tag=20260713-pureasr}
export scope=${scope:-same}
export N=${N:-1}
export max_past=${max_past:-30}
budget=${budget:-8}                       # スロット予算 M
force=${force:-0}                         # 1 で学習済みも再学習
POOLS=${POOLS:-"attn frames conv query topk utt"}
train_set=${train_set:-train_nodup}
variant="${scope}_n${N}_p${max_past}"

# --- 平均過去長から conv の圧縮率 c を決める ------------------------------
# フレーム長 = hop 132 サンプル × conv2d 1/4 サブサンプリング = 528 サンプル/フレーム
bounds="dump/raw/${train_set}/past_speech_${variant}_bounds.scp"
if [ ! -f "${bounds}" ]; then
    echo "[compare] past_speech 生成: ${train_set} (${variant})"
    $PY local/build_past_audio.py --seg_dir "data/${train_set}" --dump_dir "dump/raw/${train_set}" \
        --scope "${scope}" --n_past "${N}" --max_past_sec "${max_past}" \
        --out_name "past_speech_${variant}"
fi
read -r mean_sec mean_tp conv_c < <(
    awk -v m="${budget}" '{s=0; for(i=2;i<=NF;i++) s+=$i; t+=s; n++}
        END{tp=t/n/528; c=int(tp/m+0.5); if(c<1)c=1; printf "%.2f %.0f %d\n", t/n/16000, tp, c}' "${bounds}"
)

echo "================================================================"
echo " 過去要約の比較  scope=${scope}  N=${N}  max_past=${max_past}s  予算 M=${budget}"
echo " 平均過去長 ${mean_sec} s → 平均 Tp ≈ ${mean_tp} フレーム"
echo " conv の圧縮率 c = round(${mean_tp} / ${budget}) = ${conv_c}  → 平均 Np ≈ $((mean_tp / conv_c)) スロット"
echo " 対象: ${POOLS}"
echo " Stage1: ${pureasr_tag}   force=${force}"
echo "================================================================"
if [ "${conv_c}" = "1" ]; then
    echo "[注意] c=1 は圧縮なし（frames と同一）。予算 M が平均 Tp に近すぎます。M を小さくするか N を増やしてください。"
fi

for pool in ${POOLS}; do
    case "${pool}" in
        conv)         pooltag="conv_c${conv_c}";      slotinfo="平均 Np ≈ $((mean_tp / conv_c))" ;;
        query|topk)   pooltag="${pool}_m${budget}";   slotinfo="Np = ${budget}" ;;
        utt)          pooltag="utt_k1";               slotinfo="Np = 有効発話数（≤ ${N}）" ;;
        max|attn)     pooltag="${pool}";              slotinfo="Np = 1" ;;
        frames)       pooltag="frames";               slotinfo="Np = Tp ≈ ${mean_tp}" ;;
        *) echo "[skip] 未知の pool: ${pool}" >&2; continue ;;
    esac

    # 学習済み（ave チェックポイントあり）ならスキップ
    d=$(ls -d exp/asr_*-turntaking-xfmr-pool_"${pooltag}"-"${variant}" 2>/dev/null | tail -1 || true)
    if [ "${force}" != "1" ] && [ -n "${d}" ] && [ -f "${d}/valid.tag_acc.ave.pth" ]; then
        echo "===== [skip] ${pool} (${slotinfo}) は学習済み: ${d}"
        continue
    fi

    echo "===================== ${pool}  ${slotinfo}  (${variant}) ====================="
    ( export pool ratio="${conv_c}" slots="${budget}" uslots=1; ./run_xfmr_pool.sh "$@" )
done

echo "===== 比較用の学習が完了（scope=${scope} N=${N} M=${budget}）====="
echo "集計: POOLS=\"${POOLS}\" budget=${budget} N=${N} scope=${scope} ./eval_xfmr_pool_compare.sh"
