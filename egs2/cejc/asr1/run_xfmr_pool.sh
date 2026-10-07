#!/usr/bin/env bash
# Stage2（凍結Enc・LoRA/Adapterなし）で、過去発話の要約を max / attention pooling で比較する。
# 過去音声 past_speech（過去N発話を連結）を凍結エンコーダで符号化し、pool で1本に要約 → ヘッドへ。
#   pool=attn scope=same N=1 max_past=30 pureasr_tag=20260713-pureasr ./run_xfmr_pool.sh
#   pool=max  scope=same N=1 max_past=30 pureasr_tag=20260713-pureasr ./run_xfmr_pool.sh
#   pool=conv  ratio=8 scope=same N=1 ... ./run_xfmr_pool.sh   # 固定率・soft（Compressive 型）
#   pool=query slots=8 scope=same N=1 ... ./run_xfmr_pool.sh   # 固定長・soft（PMA/Perceiver 型）
#   pool=topk  slots=8 scope=same N=1 ... ./run_xfmr_pool.sh   # 固定長・hard（重要度上位を選択）
set -e; set -u; set -o pipefail
PY=${PY:-/home/kobori/.conda/envs/espnet/bin/python}
pool=${pool:?pool=max または pool=attn を指定}
N=${N:-1}; max_past=${max_past:-30}; scope=${scope:-same}
ratio=${ratio:-8}          # pool=conv の圧縮率（過去 Tp フレーム → ceil(Tp/ratio) スロット）
slots=${slots:-8}          # pool=query / topk のスロット数 M（過去長に依らず固定）
uslots=${uslots:-1}        # pool=utt の 1 発話あたりスロット数
pureasr_tag=${pureasr_tag:?Stage1のタグ 例) pureasr_tag=20260713-pureasr}
S1=exp/asr_${pureasr_tag}; S1_MODEL=${S1_MODEL:-${S1}/valid.loss.ave.pth}
train_set=train_nodup; valid_set=train_dev; test_sets="eval"

variant="${scope}_n${N}_p${max_past}"           # past_speech のデータ名（圧縮の設定に依らない）
# 圧縮系は設定ごとに別モデルなので、タグ・stats に圧縮率/スロット数を付ける
case "${pool}" in
    conv)        pooltag="conv_c${ratio}"    ;;
    query|topk)  pooltag="${pool}_m${slots}" ;;
    utt)         pooltag="utt_k${uslots}"    ;;
    *)           pooltag="${pool}"           ;;
esac
# base_cfg / tag_suffix を差し替えると PEFT 版（LoRA / Adapter）に流用できる。
# 既定は完全凍結の pool 用 config（run_xfmr_n0.sh と同じ規約）。
base_cfg=${base_cfg:-myconf/train_asr_turntaking_xfmr_pool_${pool}.yaml}
tag_suffix=${tag_suffix:-}
asr_tag=${asr_tag:-$(date +%Y%m%d)-turntaking-xfmr-pool_${pooltag}${tag_suffix}-${scope}_n${N}_p${max_past}}
# stats は過去データと形状だけで決まるので PEFT 版とも共有する（tag_suffix を付けない）
# tail=1 のときは tail_speech の形状も入るので別ディレクトリにする
tail=${tail:-0}             # 1 で無音込み学習用の tail_speech.scp を配線する（tt_tail_sec>0 のモデル用）
asr_stats_dir=${asr_stats_dir:-exp/asr_stats_raw_jp_word_pool_${pooltag}_${variant}$([ "${tail}" = "1" ] && echo _tail)}
inference_config=${inference_config:-myconf/decode_cbs_transducer_bounded.yaml}

# (1) past_speech（過去N発話の連結音声）を用意し、past_speech.scp に配置。past_vec は使わない。
for dset in "${train_set}" "${valid_set}" ${test_sets}; do
    src="dump/raw/${dset}/past_speech_${variant}.scp"
    srcb="dump/raw/${dset}/past_speech_${variant}_bounds.scp"
    # 境界ファイル（発話単位圧縮に必要）が無い旧データは作り直す
    if [ ! -f "${src}" ] || [ ! -f "${srcb}" ]; then
        echo "[run_xfmr_pool] past_speech 生成: ${dset} (${variant})"
        $PY local/build_past_audio.py --seg_dir "data/${dset}" --dump_dir "dump/raw/${dset}" \
            --scope "${scope}" --n_past "${N}" --max_past_sec "${max_past}" \
            --out_name "past_speech_${variant}"
    fi
    cp "${src}" "dump/raw/${dset}/past_speech.scp"
    cp "${srcb}" "dump/raw/${dset}/past_bounds.scp"
    # past_vec は無効化（誤配線防止）
    [ -f "dump/raw/${dset}/past_vec.scp" ] && mv "dump/raw/${dset}/past_vec.scp" "dump/raw/${dset}/past_vec.scp.off" || true
    # 無音込み学習：tail=1 のときだけ tail_speech.scp を配線する（local/build_tail_audio.py で生成）。
    # asr.sh はファイルの有無で自動的に渡すので、使わない学習では .off に退避しておく。
    if [ "${tail}" = "1" ]; then
        if [ ! -f "dump/raw/${dset}/tail_speech.scp" ]; then
            [ -f "dump/raw/${dset}/tail_speech.scp.off" ] && mv "dump/raw/${dset}/tail_speech.scp.off" "dump/raw/${dset}/tail_speech.scp" \
                || { echo "dump/raw/${dset}/tail_speech.scp がありません（local/build_tail_audio.py --dset ${dset}）"; exit 1; }
        fi
    else
        [ -f "dump/raw/${dset}/tail_speech.scp" ] && mv "dump/raw/${dset}/tail_speech.scp" "dump/raw/${dset}/tail_speech.scp.off" || true
    fi
done

# (2) init_param を Stage1 に差し替えた config を生成
cfg=myconf/.gen_pool_${pooltag}${tag_suffix}_${variant}.yaml
sed "s#exp/asr_PUREASR_TAG/valid.loss.ave.pth#${S1_MODEL}#" "${base_cfg}" > "$cfg"
# base_cfg を差し替えたときも過去要約が ${pool} と食い違わないようにそろえる
sed -i "s/^    tt_past_pool:.*/    tt_past_pool: ${pool}/" "$cfg"
case "${pool}" in
    conv)       sed -i "s/^    tt_compress_ratio: [0-9]*/    tt_compress_ratio: ${ratio}/" "$cfg" ;;
    query|topk) sed -i "s/^    tt_compress_slots: [0-9]*/    tt_compress_slots: ${slots}/" "$cfg" ;;
    utt)        sed -i "s/^    tt_utt_slots: [0-9]*/    tt_utt_slots: ${uslots}/" "$cfg" ;;
esac
echo "[run_xfmr_pool] config=${cfg} (pool=${pooltag} / init_param=${S1_MODEL})"

# (3) 学習＋デコード（ASR は凍結＝CER は Stage1 と同一）
./asr.sh --stage 10 --asr_stats_dir "${asr_stats_dir}" --ngpu 1 \
    --use_streaming false --use_disfluency_detection false \
    --use_turntaking_detection true --use_multitask_transducer false \
    --nj 16 --inference_nj 16 --lang jp --feats_type raw --token_type word \
    --lm_config conf/train_lm.yaml --asr_config "${cfg}" --asr_tag "${asr_tag}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" --valid_set "${valid_set}" --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" --use_lm false --use_word_lm false \
    --inference_asr_model valid.loss.ave.pth "$@"
echo "===== 完了: exp/${asr_tag}  評価: pool=${pool} scope=${scope} N=${N} ./eval_xfmr_pool.sh ====="
