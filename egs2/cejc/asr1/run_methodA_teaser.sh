#!/usr/bin/env bash
# 手法A（TEASER）実行スクリプト
#   設計: cif/method_A_TEASER.md（TEASER slave/master 2 段階学習）
#
# データ準備・特徴量抽出・Stage 1 学習・標準デコードは すべて asr.sh が担当する。
# このスクリプトは asr.sh を呼び出した上で、Stage 2 学習（stop_head）と
# TEASER タグ早期確定評価（dev sweep → eval）を後段に重ねるだけ。
# データが変わったら asr.sh の該当ステージ（Stage 1 データ準備, Stage 3 特徴抽出 …）
# を回せば再現できる。
#
# 使い方:
#   ./run_methodA_teaser.sh                                        # 全フェーズ
#   run_stage2=false run_sweep=false run_eval=false \
#     ./run_methodA_teaser.sh --stop_stage 5                       # データ準備+特徴抽出のみ（データ更新時）
#   run_stage2=false run_sweep=false run_eval=false \
#     ./run_methodA_teaser.sh --stage 11 --stop_stage 11           # Stage 1 学習のみ
#   run_stage1=false run_sweep=false run_eval=false \
#     ./run_methodA_teaser.sh                                       # Stage 2 学習のみ
#   run_stage1=false run_stage2=false run_eval=false \
#     ./run_methodA_teaser.sh                                       # dev sweep のみ
#   run_stage1=false run_stage2=false run_sweep=false \
#     best_threshold=0.3 best_v=1 ./run_methodA_teaser.sh          # eval のみ（sweep 済みの場合）
#
# 注: --stage / --stop_stage などの引数は Stage 1 の asr.sh にそのまま渡される。
set -e
set -u
set -o pipefail

# ===== データ =====
train_set=train_nodup
valid_set=train_dev
test_sets="eval"

# ===== 設定ファイル =====
stage1_config=myconf/train_cif_transformer.yaml
stage2_config=myconf/train_cif_teaser_stage2.yaml
inference_config=myconf/decode_transformer.yaml
lm_config=conf/train_lm.yaml

# ===== 実験タグ =====
# v2: クラス重み [1.92, 1.00, 1.63]、位置重み廃止、cif_proj(256→16) 追加
stage1_tag=cif-teaser-methodA-v2-stage1
stage2_tag=cif-teaser-methodA-v2-stage2

# ===== トグル（env で上書き可）=====
run_stage1="${run_stage1:-true}"   # Stage 1: asr.sh（データ前処理〜tag_classifier 学習〜標準デコード）
run_stage2="${run_stage2:-true}"   # Stage 2: stop_head 学習（tag_classifier は freeze）
run_sweep="${run_sweep:-true}"     # dev sweep: stop_threshold × v の grid search
run_eval="${run_eval:-true}"       # eval: eval セットでの最終評価

# ===== TEASER 評価設定 =====
threshold_sweep="${threshold_sweep:-0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
v_sweep="${v_sweep:-1 2 3 4 5}"
# dev sweep の結果を確認後に設定する（method_A_TEASER.md §7 参照）
best_threshold="${best_threshold:-0.5}"
best_v="${best_v:-2}"

# ===== パス =====
stage1_expdir=exp/asr_${stage1_tag}
stage2_expdir=exp/asr_${stage2_tag}
stage1_model=${stage1_expdir}/valid.cer.best.pth
stage2_model=${stage2_expdir}/valid.loss.best.pth
stage2_config_saved=${stage2_expdir}/config.yaml
asr_stats_dir=exp/asr_stats_raw_jp_word   # Stage 2 で Stage 1 の stats を再利用

# ------------------------------------------------------------------
# 1. asr.sh: データ準備 → 特徴量抽出 → Stage 1 学習（tag_classifier）→ 標準デコード
#    --stage / --stop_stage などの引数は "$@" 経由でそのまま asr.sh に渡る。
# ------------------------------------------------------------------
if "${run_stage1}"; then
    ./asr.sh                                               \
        --ngpu 1                                           \
        --use_streaming false                              \
        --use_multitask_transducer false                   \
        --use_disfluency_detection false                   \
        --use_cif_detection true                           \
        --nj 16                                            \
        --inference_nj 16                                  \
        --lang jp                                          \
        --feats_type raw                                   \
        --token_type word                                  \
        --lm_config "${lm_config}"                         \
        --asr_config "${stage1_config}"                    \
        --asr_tag "${stage1_tag}"                          \
        --inference_config "${inference_config}"           \
        --train_set "${train_set}"                         \
        --valid_set "${valid_set}"                         \
        --test_sets "${test_sets}"                         \
        --lm_train_text "data/${train_set}/text"           \
        --use_lm false                                     \
        --use_word_lm false                                \
        --inference_asr_model valid.cer.best.pth           \
        "$@"
fi

# ------------------------------------------------------------------
# 2. Stage 2 学習: stop_head のみを BCE 損失で学習（tag_classifier は freeze）
#    asr.sh --stage 11 --stop_stage 11 で学習ステージだけを実行する。
#    stats は Stage 1 と同一データなので再収集不要（--asr_stats_dir で指定）。
# ------------------------------------------------------------------
if "${run_stage2}"; then
    if [ ! -f "${stage1_model}" ]; then
        echo "[run_methodA_teaser] Stage 1 チェックポイントが見つかりません: ${stage1_model}" >&2
        echo "  先に Stage 1 を実行（run_stage1=true）するか、" >&2
        echo "  stage1_tag に学習済みの exp タグを指定してください。" >&2
        exit 1
    fi

    ./asr.sh                                               \
        --stage 11 --stop_stage 11                         \
        --ngpu 1                                           \
        --use_streaming false                              \
        --use_multitask_transducer false                   \
        --use_disfluency_detection false                   \
        --use_cif_detection true                           \
        --nj 16                                            \
        --lang jp                                          \
        --feats_type raw                                   \
        --token_type word                                  \
        --lm_config "${lm_config}"                         \
        --asr_config "${stage2_config}"                    \
        --asr_tag "${stage2_tag}"                          \
        --asr_stats_dir "${asr_stats_dir}"                 \
        --pretrained_model "${stage1_model}"               \
        --ignore_init_mismatch true                        \
        --train_set "${train_set}"                         \
        --valid_set "${valid_set}"                         \
        --test_sets "${test_sets}"                         \
        --lm_train_text "data/${train_set}/text"           \
        --use_lm false                                     \
        --use_word_lm false
fi

# ------------------------------------------------------------------
# 3. dev sweep: stop_threshold × v の grid search（HM 最大の組み合わせを選ぶ）
# ------------------------------------------------------------------
if "${run_sweep}"; then
    if [ ! -f "${stage2_model}" ]; then
        echo "[run_methodA_teaser] Stage 2 チェックポイントが見つかりません: ${stage2_model}" >&2
        echo "  先に Stage 2 学習（run_stage2=true）を実行してください。" >&2
        exit 1
    fi

    sweep_log="${stage2_expdir}/dev_sweep_teaser.log"
    mkdir -p "${stage2_expdir}"
    echo "============================================================"  | tee -a "${sweep_log}"
    echo " dev sweep: stop_threshold × v"                               | tee -a "${sweep_log}"
    echo " threshold: ${threshold_sweep}"                                | tee -a "${sweep_log}"
    echo " v        : ${v_sweep}"                                        | tee -a "${sweep_log}"
    echo " 開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"                      | tee -a "${sweep_log}"
    echo "============================================================"  | tee -a "${sweep_log}"

    for thr in ${threshold_sweep}; do
        for v in ${v_sweep}; do
            echo "----- stop_threshold=${thr}  v=${v} -----" | tee -a "${sweep_log}"
            python local/tag_eval.py                           \
                --method         teaser                        \
                --config         "${stage2_config_saved}"      \
                --model          "${stage2_model}"             \
                --wav_scp        dump/raw/${valid_set}/wav.scp \
                --tag            dump/raw/${valid_set}/tag     \
                --stop_threshold "${thr}"                      \
                --teaser_v       "${v}"                        \
                2>&1 | tee -a "${sweep_log}"
        done
    done

    echo ""                                                              | tee -a "${sweep_log}"
    echo " 終了時刻: $(date '+%Y-%m-%d %H:%M:%S')"                      | tee -a "${sweep_log}"
    echo "sweep ログ: ${sweep_log}"
    echo ""
    echo "HM が最大の (threshold, v) を確認して以下のように再実行してください:"
    echo "  run_stage1=false run_stage2=false run_sweep=false \\"
    echo "    best_threshold=<値> best_v=<値> ./run_methodA_teaser.sh"
    echo ""
fi

# ------------------------------------------------------------------
# 4. eval: eval セットでの最終評価
# ------------------------------------------------------------------
if "${run_eval}"; then
    if [ ! -f "${stage2_model}" ]; then
        echo "[run_methodA_teaser] Stage 2 チェックポイントが見つかりません: ${stage2_model}" >&2
        echo "  先に Stage 2 学習（run_stage2=true）を実行してください。" >&2
        exit 1
    fi

    eval_log="${stage2_expdir}/eval_teaser_thr${best_threshold}_v${best_v}.log"
    echo "============================================================"  | tee "${eval_log}"
    echo " eval: stop_threshold=${best_threshold}  v=${best_v}"         | tee -a "${eval_log}"
    echo " 開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"                      | tee -a "${eval_log}"
    echo "============================================================"  | tee -a "${eval_log}"
    python local/tag_eval.py                                   \
        --method         teaser                                \
        --config         "${stage2_config_saved}"              \
        --model          "${stage2_model}"                     \
        --wav_scp        dump/raw/eval/wav.scp                 \
        --tag            dump/raw/eval/tag                     \
        --stop_threshold "${best_threshold}"                   \
        --teaser_v       "${best_v}"                           \
        --output         "${stage2_expdir}/tag_eval_teaser_thr${best_threshold}_v${best_v}.tsv" \
        2>&1 | tee -a "${eval_log}"

    echo "結果ログ: ${eval_log}"
    echo "結果 TSV: ${stage2_expdir}/tag_eval_teaser_thr${best_threshold}_v${best_v}.tsv"
fi
