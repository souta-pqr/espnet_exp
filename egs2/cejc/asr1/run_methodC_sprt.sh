#!/usr/bin/env bash
# 手法C（SPRT）実行スクリプト
#   設計: cif/sprt_impl.md（Wald 1945 の SPRT 枠組みを借用）
#
# データ準備・特徴量抽出・学習・標準デコードは すべて asr.sh が担当する。
# このスクリプトは asr.sh を呼び出した上で、SPRT のタグ早期確定評価
# （dev sweep → eval）を後段に重ねるだけ。データが変わったら asr.sh の
# 該当ステージ（Stage1 データ準備, Stage3 特徴抽出 …）を回せば再現できる。
#
# 使い方:
#   ./run_methodC_sprt.sh                                   # asr.sh 全ステージ + SPRT 評価
#   run_tag_eval=false ./run_methodC_sprt.sh --stop_stage 5 # データ準備+特徴抽出のみ（データ更新時）
#   run_tag_eval=false ./run_methodC_sprt.sh --stage 11 --stop_stage 11   # 学習のみ
#   run_asr=false best_upper=1.5 ./run_methodC_sprt.sh      # 学習済みで SPRT 評価のみ
#
# 注: --stage / --stop_stage などの引数は そのまま asr.sh に渡される。
set -e
set -u
set -o pipefail

# ===== データ =====
train_set=train_nodup
valid_set=train_dev
test_sets="eval"

# ===== 設定ファイル =====
asr_config=myconf/train_cif_transformer.yaml
inference_config=myconf/decode_transformer.yaml
lm_config=conf/train_lm.yaml

# ===== 実験タグ（exp/asr_${asr_tag}）=====
asr_tag=cif-sprt-methodC

# ===== トグル（env で上書き可）=====
run_asr="${run_asr:-true}"              # asr.sh（データ準備〜学習〜標準デコード）を回すか
run_tag_eval="${run_tag_eval:-true}"    # SPRT タグ評価（dev sweep + eval）を回すか

# ===== SPRT 評価設定 =====
sprt_sweep="${sprt_sweep:-0.5 1.0 1.5 2.0 3.0 5.0}"   # dev sweep の探索範囲（sprt_impl.md §6）
chunk_frames="${chunk_frames:-76}"      # streaming のチャンク幅（特徴量フレーム数）
best_upper="${best_upper:-2.0}"         # eval で使う境界。dev sweep の HM 最大値を入れる

expdir=exp/asr_${asr_tag}
model=${expdir}/valid.cer.best.pth
config=${expdir}/config.yaml

# ------------------------------------------------------------------
# 1. asr.sh：データ準備 → 特徴量抽出 → 学習 → 標準デコード/スコアリング
#    （手法C には追加学習ヘッドがないため通常の cif_tag 学習）
#    --stage / --stop_stage 等の引数は "$@" 経由でそのまま asr.sh に渡る。
# ------------------------------------------------------------------
if "${run_asr}"; then
    ./asr.sh                                               \
        --ngpu 1                                           \
        --use_streaming false                              \
        --use_multitask_transducer false                  \
        --use_disfluency_detection false                  \
        --use_cif_detection true                           \
        --nj 16                                            \
        --inference_nj 16                                  \
        --lang jp                                          \
        --feats_type raw                                   \
        --token_type word                                  \
        --lm_config "${lm_config}"                         \
        --asr_config "${asr_config}"                       \
        --asr_tag "${asr_tag}"                             \
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
# 2. SPRT タグ早期確定評価
#    tag_eval.py は asr.sh が生成した dump/raw/<set>/{wav.scp,tag} を使う。
# ------------------------------------------------------------------
if "${run_tag_eval}"; then
    if [ ! -f "${model}" ]; then
        echo "[run_methodC_sprt] 学習済みモデルが見つかりません: ${model}" >&2
        echo "  先に学習（run_asr=true）を実行してください。" >&2
        exit 1
    fi

    # --- 2a. dev set で sprt_upper を sweep（HM 最大の境界を選ぶ）---
    echo "============================================================"
    echo " dev sweep over sprt_upper  (各出力末尾の HM が最大の値を best_upper に)"
    echo "============================================================"
    for upper in ${sprt_sweep}; do
        echo "----- sprt_upper=${upper} -----"
        python local/tag_eval.py                       \
            --config  "${config}"                      \
            --model   "${model}"                       \
            --wav_scp dump/raw/${valid_set}/wav.scp    \
            --tag     dump/raw/${valid_set}/tag        \
            --sprt_upper "${upper}"
    done

    # --- 2b. eval set で最終評価（batch / streaming）---
    echo "============================================================"
    echo " eval (batch)  sprt_upper=${best_upper}"
    echo "============================================================"
    python local/tag_eval.py                           \
        --config  "${config}"                          \
        --model   "${model}"                           \
        --wav_scp dump/raw/eval/wav.scp                \
        --tag     dump/raw/eval/tag                    \
        --sprt_upper "${best_upper}"                   \
        --output  "${expdir}/tag_eval_sprt${best_upper}.tsv"

    echo "============================================================"
    echo " eval (streaming)  sprt_upper=${best_upper}  chunk=${chunk_frames}"
    echo "============================================================"
    python local/tag_eval.py                           \
        --config  "${config}"                          \
        --model   "${model}"                           \
        --wav_scp dump/raw/eval/wav.scp                \
        --tag     dump/raw/eval/tag                    \
        --streaming --chunk_frames "${chunk_frames}"   \
        --sprt_upper "${best_upper}"                   \
        --output  "${expdir}/tag_eval_sprt${best_upper}_streaming.tsv"
fi
