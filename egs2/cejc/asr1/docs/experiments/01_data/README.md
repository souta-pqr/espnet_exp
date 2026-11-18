# 01 データの準備

CEJC を話者が重ならない学習・開発・評価に分け，ESPnet の `data/` と `dump/` を作る．
ここで作るものはどの実験でも使う．

[← 全体の README](../../../../../../README.md)

## できるもの

| 名前 | 中身 | 使う実験 |
| --- | --- | --- |
| `data/{train_g,dev_g,eval_g}` | 話者が重ならない分割（評価 5 グループ・開発 5 グループ，残りが学習） | すべて |
| `data/train_g_extra` | 分割の時に漏れていた 8 会話（すべて学習側） | `train_gx` |
| `data/train_gx` | `train_g` ＋ `train_g_extra` | 04・05・06 の学習 |
| `dump/raw/<set>/tail_speech.scp` | 区間末の後の音声（今は 2 秒） | 04 |
| `dump/raw/<set>/past_speech.scp` | 同じ話者の直前の発話の音声 | 04（過去の文脈を入れる版） |
| `dump/raw/train_gxm` | 「前の区間 間 今の区間」をつないだ区間を足したもの | 04（gtm 系）・`train_gxac` の元 |
| `data/train_gxac` | `train_gx` ＋ つないだ区間 | 05 の提案手法 |

## 手順

作業ディレクトリは `egs2/cejc/asr1`．

### 1. 分割して `data/` を作る

```bash
python local/cejc_groupsplit.py --seed 0          # data/{train_g,dev_g,eval_g}
python local/cejc_groupsplit_extra.py              # data/train_g_extra
```

- ラベル（`data/*/tag`）は，このときは元の Qwen3-8B のもの．[02 ラベル](../02_labels/README.md) で付け直す．
- 発話 ID と LLM の判定の対応は，番号ではなく時刻で付ける（`local/turntaking/gold1000_map.py`）．

### 2. 速度摂動・dump・トークン一覧（Stage1 の準備）

```bash
setsid nohup bash tt_batch_logs/run_gsplit_data.sh >> tt_batch_logs/gsplit_data.log 2>&1 < /dev/null &
```

`asr.sh` の stage 2〜5 と 10 を流す．dump は約 25〜30 GB．GPU の学習と並行させると I/O で遅くなるので，学習が無いときに流す．

### 3. `train_gx` と区間末の後の音声

`tt_batch_logs/run_g_stage2.sh` の段 2（`train_g_extra` の音声の書き出し → Stage1 でデコード → `train_gx` を組む）で作る．
区間末の後の音声は次で作り直せる（04 の gt 以降は 2 秒）．

```bash
python local/build_tail_audio.py --dset train_gx --tail_sec 2.0 --nj 8
python local/build_tail_audio.py --dset dev_g    --tail_sec 2.0 --nj 8
```

過去の発話の音声（同じ話者の直前 5 発話，30 秒以内）は次で作り，`past_speech.scp`・`past_bounds.scp` に写す（`run_g_stage2.sh` の段 4 と同じ）．

```bash
python local/build_past_audio.py --seg_dir data/<set> --dump_dir dump/raw/<set> \
    --scope same --n_past 5 --max_past_sec 30 --out_name past_speech_same_n5_p30
cp dump/raw/<set>/past_speech_same_n5_p30.scp dump/raw/<set>/past_speech.scp
cp dump/raw/<set>/past_speech_same_n5_p30_bounds.scp dump/raw/<set>/past_bounds.scp
```

### 4. 前の区間とつないだ音声 `train_gxm`

```bash
python local/make_gxm.py        # → dump/raw/train_gxm と統計 exp/asr_stats_..._tail_gxm
```

- 継続の区間の後に，同じ話者の区間が 0.4〜3 秒の間で続く組（58,617 組）ごとに，「前の区間 間 今の区間」を 1 つの学習例として足す．ラベルと区間末の後の音声は今の区間のもの．
- ディスクに書くのは間の音声だけ（約 1 GB）．読むときに `concat_sound`（`asr_gxm.sh` の `SPEECH_TYPE`）で 3 つをつなぐ．

### 5. 提案手法の学習データ `train_gxac`

```bash
python local/encac/make_train_gxac.py
```

- `train_gx` の全区間に，`dump/raw/train_gxm` のつないだ区間（ID が `m` で終わるもの，58,617 区間）を足す．計 375,612 区間．
- 先に 4 の `train_gxm` を作っておく．

## 注意

- `data/`・`dump/` は CEJC の書き起こしと音声を含むので，git に入れない．
- 発話の長さで除外されるため，`dump/raw/train_g` は `data/train_g` より少し少ない．
