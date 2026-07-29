#!/usr/bin/env bash
# Stage2 の「真の N=0」: 凍結エンコーダ＋Self-Attention で、過去ベクトルを一切使わずに学習。
# N=1..5 と同一アーキテクチャなので、差分は「過去発話の有無」だけになる。
#   pureasr_tag=20260713-pureasr ./run_xfmr_n0.sh
set -e; set -u; set -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
pureasr_tag=${pureasr_tag:?Stage1のタグを指定してください 例) pureasr_tag=20260713-pureasr}
S1=exp/asr_${pureasr_tag}
S1_MODEL=${S1_MODEL:-${S1}/valid.loss.ave.pth}
train_set=train_nodup; valid_set=train_dev; test_sets="eval"

# base_cfg / tag_suffix を差し替えると LoRA 版などに流用できる（既定は完全凍結）
base_cfg=${base_cfg:-myconf/train_asr_turntaking_xfmr.yaml}
tag_suffix=${tag_suffix:-}
variant="n0_xfmr${tag_suffix}"
asr_tag=${asr_tag:-$(date +%Y%m%d)-turntaking-xfmr${tag_suffix}-n0}
asr_stats_dir=exp/asr_stats_raw_jp_word_${variant}
inference_config=${inference_config:-myconf/decode_cbs_transducer_bounded.yaml}

# (1) 過去入力を一切渡さない（past_vec.scp があると asr.sh が自動で配線してしまう）
for dset in "${train_set}" "${valid_set}" ${test_sets}; do
    for f in past_vec past_speech ctx_vec; do
        if [ -f "dump/raw/${dset}/${f}.scp" ]; then
            mv "dump/raw/${dset}/${f}.scp" "dump/raw/${dset}/${f}.scp.off"
            echo "[run_xfmr_n0] 退避: dump/raw/${dset}/${f}.scp -> .off"
        fi
    done
done

# (2) init_param を Stage1 モデルに差し替えた config を生成（N>=1 と同一設定）
cfg=myconf/.gen_train_asr_turntaking_xfmr_${variant}.yaml
sed "s#exp/asr_PUREASR_TAG/valid.loss.ave.pth#${S1_MODEL}#" "${base_cfg}" > "$cfg"
echo "[run_xfmr_n0] config=${cfg} (base=${base_cfg} / init_param=${S1_MODEL} / past_vec なし)"

# (3) collect-stats → 学習 → デコード
./asr.sh --stage 10 --asr_stats_dir "${asr_stats_dir}" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml --asr_config "${cfg}" --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth "$@"

echo
echo "===== N=0 学習完了: exp/${asr_tag} ====="
echo "評価:  ./eval_xfmr_n0.sh"
