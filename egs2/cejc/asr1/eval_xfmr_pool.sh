#!/usr/bin/env bash
# 過去音声プーリング版（max/attn）のタグ性能を評価。
#   pool=attn scope=same N=1 max_past=30 ./eval_xfmr_pool.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
pool=${pool:?pool=max または attn}; N=${N:-1}; max_past=${max_past:-30}; scope=${scope:-same}
ratio=${ratio:-8}; slots=${slots:-8}; uslots=${uslots:-1}   # conv圧縮率 / query,topkスロット / uttスロット
ckpt=${ckpt:-valid.tag_acc.ave.pth}
case "${pool}" in
    conv)        pooltag="conv_c${ratio}"    ;;
    query|topk)  pooltag="${pool}_m${slots}" ;;
    utt)         pooltag="utt_k${uslots}"    ;;
    *)           pooltag="${pool}"           ;;
esac
dset=${dset:-eval}          # eval（本番）/ train_dev（閾値較正用）
dump=${dump:-1}             # 1 で区間ごとの確率を TSV にダンプ（AUC・較正をオフライン計算する）
# tt_use_dec のモデルにデコーダへ渡すトークン列。学習は正解文だが、推論で正解文を渡すと
# ラベル生成に使われた入力をそのまま見せることになり、他手法（音声のみ）と比較できない。
#   dec_text=asr （既定）… ASR 仮説。実運用条件。主結果はこちら
#   dec_text=ref       … 正解書き起こし。理想カスケード相当の上限。参考値としてのみ
#   dec_text=none      … 渡さない（学習時と条件が食い違うので通常は使わない）
dec_text=${dec_text:-asr}
# 早期確定：現発話を「発話末の trunc 秒手前」で打ち切って評価する（過去発話は不変）。
# 0 なら従来どおり発話全体を聞いてから判定。
trunc=${trunc:-0}
# 前向きグリッド：発話開始から keep 秒だけ渡す（経過時間ベース。trunc とは排他）。
keep=${keep:-0}
variant="${scope}_n${N}_p${max_past}"
# PEFT 版（run_peft_lora.sh / run_peft_adapter.sh）は tag_suffix でタグが分かれる
tag_suffix=${tag_suffix:-}
pooltag="${pooltag}${tag_suffix}"
d=$(ls -d exp/asr_*-turntaking-xfmr-pool_${pooltag}-${scope}_n${N}_p${max_past} 2>/dev/null | tail -1 || true)
[ -n "$d" ] || { echo "学習済みモデルが見つかりません"; exit 1; }
ps="dump/raw/${dset}/past_speech_${variant}.scp"
[ -f "$ps" ] || { echo "$ps がありません"; exit 1; }
sfx=""; [ "${dset}" = "eval" ] || sfx="_${dset}"
trunc_opt=""
if [ "${keep}" != "0" ]; then
    trunc_opt="--keep_sec ${keep}"
    sfx="${sfx}_e${keep}"
    # withsil=1 で元録音から切り出し、発話後の無音まで含めて渡す（実運用条件）。
    if [ "${withsil:-0}" = "1" ]; then
        trunc_opt="${trunc_opt} --full_wav_scp data/${dset}/wav.scp --segments data/${dset}/segments"
        sfx="${sfx}s"
    fi
elif [ "${trunc}" != "0" ]; then
    trunc_opt="--trunc_sec ${trunc}"
    sfx="${sfx}_t${trunc}"
fi
out="tt_eval_logs/pool_${pooltag}_${scope}_n${N}${sfx}_${ckpt%.pth}.log"; mkdir -p tt_eval_logs
dump_opt=""
if [ "${dump}" = "1" ]; then
    mkdir -p tt_preds
    dump_opt="--dump_preds tt_preds/pool_${pooltag}_${scope}_n${N}${sfx}_${ckpt%.pth}.tsv"
fi
# tt_use_dec のモデルのときだけデコーダ用トークン列を渡す。
dec_opt=""
if grep -q "tt_use_dec: true" "$d/config.yaml" 2>/dev/null; then
    case "${dec_text}" in
        asr)
            dt=$(ls -d exp/asr_*-pureasr/decode_*/"${dset}"/text 2>/dev/null | tail -1 || true)
            [ -n "$dt" ] || { echo "ASR 仮説 text が見つかりません。dec_text=ref か none を指定してください"; exit 1; }
            ;;
        ref)  dt="data/${dset}/text" ;;
        none) dt="" ;;
        *)    echo "dec_text は asr / ref / none"; exit 1 ;;
    esac
    [ -z "$dt" ] || dec_opt="--dec_text_scp ${dt}"
    out="${out%.log}_dec-${dec_text}.log"
    [ "${dump}" = "1" ] && dump_opt="--dump_preds tt_preds/pool_${pooltag}_${scope}_n${N}${sfx}_${ckpt%.pth}_dec-${dec_text}.tsv"
    echo "  [tt_use_dec] デコーダ入力: ${dec_text} (${dt:-なし})"
fi
echo "===== eval pool=${pooltag} ${scope} N=${N} dset=${dset} (${d}) -> ${out} ====="
$PY local/turntaking/evaluate_multitask.py \
    --config "$d/config.yaml" --model "$d/$ckpt" \
    --data_dir "dump/raw/${dset}" --past_pool --past_speech_scp "$ps" ${dump_opt} ${dec_opt} ${trunc_opt} \
    > "$out" 2>&1
grep -E "macro-F1|<継続>|<終了>|<相槌>|二値分離" "$out" | sed 's/^/    /'
