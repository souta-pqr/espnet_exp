#!/usr/bin/env bash
# 実験2: speed perturbation 3-way（0.9 / 1.0 / 1.1）を blk42 の上に積む。
#
# 対照は実験1 blk42（CER 21.2 / WER 28.2 / valid 11.775）。設定差は学習データのみ
# （config の差は max_epoch 40->20 だけ。データが 3 倍で 1 エポックも 3 倍長いため）。
#
# 注意: asr.sh の stage 2 は ref_text_files_str="text" しか引き継がないので、
#       このプロジェクト独自の tag（ターンテイキングのラベル）が失われる。
#       stage 2 相当だけ自前で --utt_extra_files "text tag" 付きで実行し、
#       stage 3 以降に渡す（stage 3/4 は tag を明示コピーする実装が入っている）。
#
# cron から起動する（user@1609.service の外で走らせて systemd-oomd を避ける）。
set -u
crontab -r 2>/dev/null
HERE=/home/kobori/2025/espnet/egs2/cejc/asr1
cd "$HERE"
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH

TAG=20260921-pureasr-blk42-sp
STATS=exp/asr_stats_raw_jp_word_pureasr_sp
INIT=exp/asr_20260920-pureasr-blk42/valid.loss.best.pth

echo "===== [$(date '+%m-%d %H:%M')] 実験2 speed perturb 開始 ====="
echo "cgroup: $(cat /proc/self/cgroup)"

# ---- stage 2 相当（tag を引き継ぐ）----
if [ ! -f data/train_nodup_sp/tag ]; then
    echo "--- [$(date '+%m-%d %H:%M')] speed perturbation: data/train_nodup -> data/train_nodup_sp ---"
    dirs=""
    for f in 0.9 1.1; do
        scripts/utils/perturb_data_dir_speed.sh --utt_extra_files "text tag" \
            "$f" data/train_nodup "data/train_nodup_sp$f" || exit 1
        dirs="$dirs data/train_nodup_sp$f"
    done
    utils/combine_data.sh --extra_files "text tag" \
        data/train_nodup_sp data/train_nodup $dirs || exit 1
else
    echo "--- data/train_nodup_sp は作成済み。スキップ ---"
fi
echo "--- 行数チェック（wav.scp / text / tag が揃っているか）---"
for f in wav.scp text tag; do printf '%-9s %s\n' "$f" "$(wc -l < data/train_nodup_sp/$f)"; done

common=(--asr_stats_dir "$STATS" --ngpu 1
        --speed_perturb_factors "0.9 1.0 1.1"
        --use_streaming false --use_disfluency_detection false
        --use_turntaking_detection true --use_multitask_transducer false
        --use_context_inputs false
        --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word
        --lm_config conf/train_lm.yaml
        --asr_config myconf/train_asr_pureasr_conformer_blk42_sp.yaml
        --asr_tag "$TAG"
        --inference_config myconf/decode_cbs_transducer_bounded.yaml
        --train_set train_nodup --valid_set train_dev --test_sets eval
        --lm_train_text data/train_nodup/text --use_lm false --use_word_lm false
        --inference_asr_model valid.loss.ave.pth)

# stage 5（token_list 生成）は既存の token list を壊すので通さない
echo "===== [$(date '+%m-%d %H:%M')] stage 3-4: dump 作成 ====="
./asr.sh --stage 3 --stop_stage 4 "${common[@]}" \
  || { echo "===== [$(date '+%m-%d %H:%M')] dump 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] stage 10: collect stats ====="
./asr.sh --stage 10 --stop_stage 10 "${common[@]}" \
  || { echo "===== [$(date '+%m-%d %H:%M')] collect stats 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] stage 11: 学習 ====="
./asr.sh --stage 11 --stop_stage 11 --pretrained_model "$INIT" "${common[@]}" \
  || { echo "===== [$(date '+%m-%d %H:%M')] 学習 失敗 ====="; exit 1; }

echo "===== [$(date '+%m-%d %H:%M')] stage 12-13: eval デコード ====="
./asr.sh --stage 12 --stop_stage 13 "${common[@]}" \
  && echo "===== [$(date '+%m-%d %H:%M')] デコード 完了 =====" \
  || echo "===== [$(date '+%m-%d %H:%M')] デコード 失敗 ====="

echo "--- 対照 blk42 CER 21.2 / WER 28.2 との比較 ---"
grep -A4 "^### CER" exp/asr_$TAG/RESULTS.md 2>/dev/null | tail -1
echo "===== [$(date '+%m-%d %H:%M')] 完了 ====="
