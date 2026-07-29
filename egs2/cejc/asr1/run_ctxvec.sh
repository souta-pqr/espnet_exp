#!/usr/bin/env bash
# 過去文脈を「特徴ベクトル」として注入する turn-taking 学習（ASR は不変）。
# current 発話は tight（trailing なし）。ctx_vec を発話区間末ヘッドに concat（§ turntaking_pastcontext_method.md）。
#
# 過去文脈の作り方を context_scope で切替（モデル・配線は不変）:
#   same    : 同一話者の直前N発話の encoder 平均（既定）
#   session : 同一セッション=両話者の直前N発話の平均（相手の発話も含む）
#   cluster : 全 uvec を KMeans(K) でクラスタリングし割当重心ベクトル
#
# 使い方:
#   N=2 ./run_ctxvec.sh                                  # same, N=2
#   context_scope=session N=2 ./run_ctxvec.sh            # 両話者, N=2
#   context_scope=cluster n_clusters=50 ./run_ctxvec.sh  # クラスタリング K=50
set -e; set -u; set -o pipefail

PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
N=${N:-1}
max_past=${max_past:-10}
context_scope=${context_scope:-same}     # same | session | cluster
top_k=${top_k:-5}                        # cluster: 過去の類似発話 上位何件を集めるか
feat_source=${feat_source:-encoder}      # encoder=encoder平均(256) | raw=エンコーダ非経由の生fbank平均(80)
feat_norm=${feat_norm:-global}           # raw時のみ: global=global_mvn適用 | none=生log-mel

# ctx_vec 抽出に使う学習済み encoder（新アーキ＝実験②と一貫）
ctx_model=${ctx_model:-exp/asr_20260625-turntaking-noVA-classweight/valid.tag_acc.best.pth}
ctx_config=${ctx_config:-exp/asr_20260625-turntaking-noVA-classweight/config.yaml}

train_set=train_nodup
valid_set=train_dev
test_sets="eval"

# 方式ごとに別名で出力＝衝突しない（走行中の same N=2 も保護）
case "${context_scope}" in
    same)    variant="same_n${N}_p${max_past}";    extra="--n_past ${N} --max_past_sec ${max_past}";;
    session) variant="session_n${N}_p${max_past}"; extra="--n_past ${N} --max_past_sec ${max_past}";;
    cluster) variant="cluster_top${top_k}";
             extra="--top_k ${top_k}";;
    *) echo "unknown context_scope: ${context_scope}" >&2; exit 1;;
esac

# 生特徴量(raw)は encoder を通さない別方式＝別名・別config（ctx_dim=80）で分離
if [ "${feat_source}" = "raw" ]; then
    variant="${variant}_raw"
    extra="${extra} --feat_source raw --feat_norm ${feat_norm}"
    asr_config=myconf/train_asr_turntaking_conformer_ctxvec_raw.yaml
else
    asr_config=myconf/train_asr_turntaking_conformer_ctxvec.yaml
fi
out_name="ctx_vec_${variant}"

asr_tag=${asr_tag:-$(date +%Y%m%d)-turntaking-noVA-classweight-ctx_${variant}}
asr_stats_dir=exp/asr_stats_raw_jp_word_ctx_${variant}
inference_config=myconf/decode_cbs_transducer.yaml
lm_config=conf/train_lm.yaml

# ---- (1) ctx_vec 抽出（方式別名 → 直後に ctx_vec.scp へ複製して有効化） ----
for dset in "${train_set}" "${valid_set}" ${test_sets}; do
    if [ ! -f "dump/raw/${dset}/${out_name}.scp" ]; then
        echo "[run_ctxvec] ctx_vec 抽出: ${dset} (scope=${context_scope}, ${out_name})"
        $PY local/extract_context_vectors.py \
            --config "${ctx_config}" --model "${ctx_model}" \
            --dump_dir "dump/raw/${dset}" --seg_dir "data/${dset}" \
            --context_scope "${context_scope}" --out_name "${out_name}" ${extra}
    else
        echo "[run_ctxvec] dump/raw/${dset}/${out_name}.scp 既存。再利用。"
    fi
    # asr.sh が読む固定名 ctx_vec.scp にこの方式を割り当てる（学習用）
    cp "dump/raw/${dset}/${out_name}.scp" "dump/raw/${dset}/ctx_vec.scp"
done

# ---- (2) collect-stats 〜 学習 〜 デコード（tight セット・ctx_vec 自動配線） ----
./asr.sh                                               \
    --stage 10                                         \
    --asr_stats_dir "${asr_stats_dir}"                 \
    --ngpu 1                                           \
    --use_streaming false                              \
    --use_disfluency_detection false                   \
    --use_turntaking_detection true                    \
    --use_multitask_transducer false                   \
    --nj 16 --inference_nj 16                           \
    --lang jp --feats_type raw --token_type word        \
    --lm_config "${lm_config}"                          \
    --asr_config "${asr_config}"                        \
    --asr_tag "${asr_tag}"                              \
    --inference_config "${inference_config}"            \
    --train_set "${train_set}"                          \
    --valid_set "${valid_set}"                          \
    --test_sets "${test_sets}"                          \
    --lm_train_text "data/${train_set}/text"            \
    --use_lm false --use_word_lm false                  \
    --inference_asr_model valid.loss.ave.pth            \
    "$@"
