#!/usr/bin/env bash
# Phase 6 step4: チェックポイントの選び方を変える。
#
# これまで一貫して「代理指標で選ぶと本番指標で負ける」現象が出ている:
#   Stage1: valid loss と CER で順位が逆転
#   Stage2: valid loss_tag は 27ep で旧の 39ep 最良を超えたが macro-F1 は +0.0037 だけ
#   BERT:   単体 macro-F1 は上がったのに融合後は下がった
# 現在のフレーム確率は valid.loss.ave（総損失で選んだ 10-best 平均）から出している。
# 同じ学習で valid.tag_acc.ave（タグ精度で選んだ 10-best 平均）も保存されているので、
# **タスクに近い指標で選んだ重み**に差し替えて測る。ダンプのやり直しだけで済む。
#
# 早期確定（早さ）にも直結する。融合のテキスト側は発話末以降しか使えないため、
# 発話中の早期確定は音声モデル単独の質で決まる。音声側が良くなれば早さも伸びる。
set -u
crontab -r 2>/dev/null
cd /home/kobori/2025/espnet/egs2/cejc/asr1
export PATH=/home/kobori/.conda/envs/espnet/bin:$PATH
PY=/home/kobori/.conda/envs/espnet/bin/python
NEWD=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
OLD=frame_attn-frame2_same_n5
NEW=frame_attn-frame2-blk42_same_n5

echo "===== [$(date '+%m-%d %H:%M')] step4: tag_acc 選択の重みでダンプ ====="
tag_suffix=-frame2-blk42 pool=attn scope=same N=5 max_past=30 \
  ckpt=valid.tag_acc.ave.pth sets="eval train_dev" ./run_frame_dump.sh \
  || { echo "ダンプ失敗"; exit 1; }
for ds in eval train_dev; do
  f="tt_preds/${NEW}_${ds}_valid.tag_acc.ave.tsv"
  [ -s "$f" ] || { echo "出力なし: $f"; exit 1; }
done

echo "===== [$(date '+%m-%d %H:%M')] 音声のみ（停止規則）を新旧で比較 ====="
$PY local/turntaking/frame_rules.py --step 0.05 --ckpt valid.tag_acc.ave \
    --stems "$NEW" > tt_analysis/frame_rules_blk42_tagacc.txt 2>&1 \
  && echo "OK -> tt_analysis/frame_rules_blk42_tagacc.txt" || echo "停止規則の評価に失敗"

echo "===== [$(date '+%m-%d %H:%M')] 融合（旧 BERT = 最良構成）====="
$PY local/turntaking/frame_fusion.py --stem "$NEW" --ckpt valid.tag_acc.ave \
    --bert exp/text_bert_asr \
    --asr_text "$NEWD/eval/text" --dev_asr_text "$NEWD/train_dev_dec/text" \
    > tt_analysis/frame_fusion_blk42_tagacc.txt 2>&1 \
  && echo "OK -> tt_analysis/frame_fusion_blk42_tagacc.txt" || echo "融合に失敗"

echo "--- 比較の基準（valid.loss.ave） ---"
echo "  音声のみ 0.7354 / 融合 0.7537 / 完了τ=0.70 で 0.7485・短縮 0.537 秒"
echo "===== [$(date '+%m-%d %H:%M')] step4 完了 ====="
