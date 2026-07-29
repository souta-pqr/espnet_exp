#!/usr/bin/env bash
# 過去発話を「テキスト」で入れる版（音声は現発話のみ）。
#   scope=same N=1 max_past=30 pureasr_tag=20260713-pureasr ./run_xfmr_text.sh
set -e; set -u; set -o pipefail
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
N=${N:-1}; max_past=${max_past:-30}; scope=${scope:-same}
pureasr_tag=${pureasr_tag:?Stage1のタグ 例) pureasr_tag=20260713-pureasr}
S1=exp/asr_${pureasr_tag}; S1_MODEL=${S1_MODEL:-${S1}/valid.loss.ave.pth}
train_set=train_nodup; valid_set=train_dev; test_sets="eval"

variant="${scope}_n${N}_p${max_past}"
asr_tag=${asr_tag:-$(date +%Y%m%d)-turntaking-xfmr-text-${scope}_n${N}_p${max_past}}
asr_stats_dir=exp/asr_stats_raw_jp_word_text_${variant}
inference_config=${inference_config:-myconf/decode_cbs_transducer_bounded.yaml}

# (1) past_text（過去N発話のトークンID列）を生成し past_text.scp に配置。past_vec/past_speech は無効化。
for dset in "${train_set}" "${valid_set}" ${test_sets}; do
    src="dump/raw/${dset}/past_text_${variant}.scp"
    if [ ! -f "${src}" ]; then
        echo "[run_xfmr_text] past_text 生成: ${dset} (${variant})"
        $PY local/build_past_text.py --seg_dir "data/${dset}" --dump_dir "dump/raw/${dset}" \
            --scope "${scope}" --n_past "${N}" --max_past_sec "${max_past}" \
            --out_name "past_text_${variant}"
    fi
    cp "${src}" "dump/raw/${dset}/past_text.scp"
    for f in past_vec past_speech; do
        [ -f "dump/raw/${dset}/${f}.scp" ] && mv "dump/raw/${dset}/${f}.scp" "dump/raw/${dset}/${f}.scp.off" || true
    done
done

cfg=myconf/.gen_text_${variant}.yaml
sed "s#exp/asr_PUREASR_TAG/valid.loss.ave.pth#${S1_MODEL}#" myconf/train_asr_turntaking_xfmr_text.yaml > "$cfg"
echo "[run_xfmr_text] config=${cfg} (過去=テキスト / init_param=${S1_MODEL})"

./asr.sh --stage 10 --asr_stats_dir "${asr_stats_dir}" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml --asr_config "${cfg}" --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth "$@"
echo "===== 完了: exp/${asr_tag}  評価: scope=${scope} N=${N} ./eval_xfmr_text.sh ====="
