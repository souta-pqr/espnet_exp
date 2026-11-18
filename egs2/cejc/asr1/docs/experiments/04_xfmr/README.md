# 04 エンコーダ固定＋Transformer（ESPnet の Stage2）

[03 音声認識](../03_stage1_asr/README.md) のエンコーダを固定し，その上に 2 層の Transformer を載せて，区間末の種類を判定する．
今の報告では，これを比べる相手として使う（実時間の評価の `--model proposed_new`）．

[← 全体の README](../../../../../../README.md)

- モデル：`espnet2/asr/espnet_model_turntaking_xfmr.py`（`model: turntaking_xfmr`）
- 設定：`myconf/.gen_pool_attn-frame2-blk42xh2-<名前>_same_n5_p30.yaml`．`.gen_*` は `.gitignore` に当たるので，足すときは `git add -f`．
- 学習は `asr.sh --stage 11`（ESPnet の普通の学習）．どの版も，前の版の重みから続けて学習する．

## 版の関係

| 名前 | 何を変えたか | 初期値 | 流すスクリプト |
| --- | --- | --- | --- |
| gm | gemma のラベルで学習（読み出しは区間末 −0.6〜+0.75 秒） | ge | `tt_batch_logs/run_g_stage2gm.sh` |
| gt | 区間末の後の音声を 2 秒に広げる | gm | `tt_batch_logs/run_g_stage2gt.sh` |
| gtnopast | 過去の文脈なし | gt と同じ手順で abnp から | `tt_batch_logs/run_abgt.sh gtnopast` |
| **gtpostmnp** | 区間末の後だけで学習＋前の区間とつないだ音声（`train_gxm`）＋過去の文脈なし | gtnopast | `tt_batch_logs/run_gxm.sh gtpostmnp` |

比べる相手として使っているのは **gtpostmnp**（exp-realtime-epd の `weights/proposed_gtpostmnp.pt`）．

## 手順

### 1. 学習

```bash
setsid nohup bash tt_batch_logs/run_g_stage2gm.sh >> tt_batch_logs/g_stage2gm.log 2>&1 < /dev/null &
setsid nohup bash tt_batch_logs/run_g_stage2gt.sh >> tt_batch_logs/g_stage2gt.log 2>&1 < /dev/null &
setsid nohup bash tt_batch_logs/run_abgt.sh gtnopast >> tt_batch_logs/abgt_gtnopast.log 2>&1 < /dev/null &
setsid nohup bash tt_batch_logs/run_gxm.sh gtpostmnp >> tt_batch_logs/ab_gtpostmnp.log 2>&1 < /dev/null &
```

- 1 本ずつ流す．GPU の学習と dump を並行させない．
- `run_gxm.sh` は `asr.sh` の代わりに `asr_gxm.sh` を使う．`asr_gxm.sh` は，環境変数 `SPEECH_TYPE=concat_sound` で「前の区間 間 今の区間」の音声をつないで読む所だけが違う．
- 各スクリプトは，学習の後にフレームダンプ（`tt_preds/frame_*_post5.0.tsv`．区間の長さ＋5 秒のスコア）も作る．

### 2. 実時間のプログラムに載せる

```bash
# silver13 で
PYTHONPATH=~/2025/espnet ~/anaconda3/envs/espnet/bin/python ~/2026/work/exp-realtime-epd/tools/export_proposed.py \
    exp/asr_20261104-turntaking-xfmr-pool_attn-frame2-blk42xh2-gtpostmnp-same_n5_p30 \
    ~/2026/work/exp-realtime-epd/weights/proposed_gtpostmnp.pt
```

その後の評価は [07 実時間の評価](../07_realtime_eval/README.md)．この方式は応答するたびに発話を区切り直すので，スコアを書き出して後から決め方を当てはめることはできず，決め方ごとに実際に流す．

### 3. オフラインの評価（参考）

フレームダンプと silero-vad の確率から，cut-in rate と latency の曲線を出す．今の報告の表は実時間の評価だけを使う．

```bash
python local/turntaking/eot_eval.py --data data/eval_g \
    --dump tt_preds/frame_..._eval_g_valid.loss.ave_post5.0.tsv --name gt \
    --silero tt_analysis/vad_silero/eval_g --out tt_analysis/eot_eval_eval_g
```

`--pick_from <開発データの出力>` で，開発データで選んだ動作点を評価データで測る．

## アブレーション（ESPnet 版）

gm・gt から 1 か所だけ変えて学習する．名前を引数に渡す．

| スクリプト | 名前 | 変えたこと |
| --- | --- | --- |
| `tt_batch_logs/run_ab.sh` | abnp・abnt・abna・abnl | 過去の文脈なし・区間末の後で学習しない・音声認識の事前学習なし（学習率 2 通り） |
| `tt_batch_logs/run_abgt.sh` | gtnopast・gtnotail・gtnoinit | 上の gt 版 |

評価は `run_ab_eval.sh`・`run_abnt_eval.sh`．

## 注意

- `train_gxm` は [01 データの準備](../01_data/README.md) の 4（`local/make_gxm.py`）で作る．
- 種だけ変えた版（gts31 など），音声認識の損失を足した版（`espnet_asrw`）は採らなかったので入れていない．
