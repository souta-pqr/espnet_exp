#!/usr/bin/env bash
# Stage2: Stage1の純粋ASRエンコーダを凍結し、[過去max-poolベクトル ; 現発話encoder出力] を
# Transformer に通して「現発話の最終フレーム位置」→MLP で区間末タグを学習。
#   pureasr_tag=<Stage1のasr_tag> scope=same N=2 ./run_xfmr.sh
set -e; set -u; set -o pipefail
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
N=${N:-2}; max_past=${max_past:-30}; scope=${scope:-same}
pureasr_tag=${pureasr_tag:?Stage1のタグを指定してください 例) pureasr_tag=20260713-pureasr}
S1=exp/asr_${pureasr_tag}
S1_MODEL=${S1_MODEL:-${S1}/valid.loss.ave.pth}
train_set=train_nodup; valid_set=train_dev; test_sets="eval"

# base_cfg / tag_suffix を差し替えると LoRA など別バリアントに流用できる（既定は完全凍結 Stage2）
base_cfg=${base_cfg:-myconf/train_asr_turntaking_xfmr.yaml}
tag_suffix=${tag_suffix:-}
variant="${scope}_n${N}_p${max_past}_xfmr${tag_suffix}"
asr_tag=${asr_tag:-$(date +%Y%m%d)-turntaking-xfmr${tag_suffix}-${scope}_n${N}_p${max_past}}
asr_stats_dir=exp/asr_stats_raw_jp_word_${variant}
inference_config=${inference_config:-myconf/decode_cbs_transducer_bounded.yaml}

# (1) 過去 max-pool ベクトル列 past_vec を事前計算（凍結エンコーダで）
for dset in "${train_set}" "${valid_set}" ${test_sets}; do
    out=dump/raw/${dset}/past_vec_${variant}
    if [ ! -f "${out}.scp" ]; then
        echo "[run_xfmr] past_vec 抽出: ${dset} (${variant})"
        $PY local/extract_past_vectors.py --config "${S1}/config.yaml" --model "${S1_MODEL}" \
            --dump_dir "dump/raw/${dset}" --seg_dir "data/${dset}" \
            --scope "${scope}" --n_past "${N}" --max_past_sec "${max_past}" \
            --out_name "past_vec_${variant}" \
            --uvec_cache "dump/raw/${dset}/uvec_max_${pureasr_tag}.npz"
    else
        echo "[run_xfmr] ${out}.scp 既存。再利用。"
    fi
    cp "${out}.scp" "dump/raw/${dset}/past_vec.scp"
    # 他の過去入力は使わない（誤配線防止）
    for f in past_speech ctx_vec; do
      [ -f "dump/raw/${dset}/${f}.scp" ] && mv "dump/raw/${dset}/${f}.scp" "dump/raw/${dset}/${f}.scp.off" || true
    done
done

# (2) init_param を Stage1 モデルに差し替えた config を生成
cfg=myconf/.gen_train_asr_turntaking_xfmr_${variant}.yaml
sed "s#exp/asr_PUREASR_TAG/valid.loss.ave.pth#${S1_MODEL}#" "${base_cfg}" > "$cfg"
echo "[run_xfmr] config=${cfg} (base=${base_cfg} / init_param=${S1_MODEL})"

# (3) collect-stats → 学習 → デコード（デコードは凍結ASR＝Stage1と同じCER）
./asr.sh --stage 10 --asr_stats_dir "${asr_stats_dir}" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml --asr_config "${cfg}" --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth "$@"
