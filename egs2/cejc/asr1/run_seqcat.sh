#!/usr/bin/env bash
# seqcat 方式: 過去N発話の音声を現発話の前に連結→共有エンコーダ→現発話フレームを max pool → 区間末タグ。
# ASR は現発話のみで不変。過去は "現発話 end より前"（同一話者では直前N発話）。
#   scope=same N=2 ./run_seqcat.sh
set -e; set -u; set -o pipefail
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
N=${N:-2}; max_past=${max_past:-30}; scope=${scope:-same}
pool_range=${pool_range:-whole}       # whole=過去+現発話全体を pool / current=現発話のみ
train_set=train_nodup; valid_set=train_dev; test_sets="eval"

# past_speech は pool_range に依存しない（同じ過去音声）→ 共有名で再利用
past_variant="${scope}_n${N}_p${max_past}"
# 実験（exp/stats/tag）は pool_range で区別
variant="${scope}_n${N}_p${max_past}_seqcat_${pool_range}"
if [ "${pool_range}" = "whole" ]; then
    asr_config=myconf/train_asr_turntaking_conformer_seqcat_whole.yaml
else
    asr_config=myconf/train_asr_turntaking_conformer_seqcat.yaml
fi
asr_tag=${asr_tag:-$(date +%Y%m%d)-turntaking-noVA-classweight-ctx_${variant}}
asr_stats_dir=exp/asr_stats_raw_jp_word_ctx_${variant}
inference_config=${inference_config:-myconf/decode_cbs_transducer_bounded.yaml}
lm_config=conf/train_lm.yaml

# (1) past_speech.scp 構築（pool_range 非依存の共有名→固定名 past_speech.scp へ複製）
for dset in "${train_set}" "${valid_set}" ${test_sets}; do
    out=dump/raw/${dset}/past_speech_${past_variant}
    if [ ! -f "${out}.scp" ]; then
        echo "[run_seqcat] past_speech 構築: ${dset} (${past_variant})"
        $PY local/build_past_audio.py --seg_dir "data/${dset}" --dump_dir "dump/raw/${dset}" \
            --scope "${scope}" --n_past "${N}" --max_past_sec "${max_past}" --out_name "past_speech_${past_variant}"
    else
        echo "[run_seqcat] ${out}.scp 既存。再利用。"
    fi
    cp "${out}.scp" "dump/raw/${dset}/past_speech.scp"
    # seqcat では ctx_vec を使わない → 残っていれば退避（asr.sh の誤配線防止）
    [ -f "dump/raw/${dset}/ctx_vec.scp" ] && mv "dump/raw/${dset}/ctx_vec.scp" "dump/raw/${dset}/ctx_vec.scp.bak" || true
done

# (2) collect-stats 〜 学習 〜 デコード（bounded 探索でメモリ暴走回避）
./asr.sh --stage 10 --asr_stats_dir "${asr_stats_dir}" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config "${lm_config}" --asr_config "${asr_config}" --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth "$@"
