# 05 提案手法

エンコーダ（contextual block conformer）＋フレームの平均＋線形層で，完了・継続・相槌の確率を出す．
エンコーダも一緒に学習し，音声認識を補助タスクとしたマルチタスク学習（CTC 損失）を足す．
実時間では，今の音声区間（silero-vad が検出した，0.4 秒以上の無音で区切られた発声の区間）だけを入れる．

[← 全体の README](../../../../../../README.md)

- コード：`local/encac/`（silver13 では `~/2026/work/easy_turn/cejc/`）
  - `enc_plug.py`：エンコーダ（環境変数 `ENC` で選ぶ）＋平均＋線形層
  - `train_ac.py`：学習．`train_et_data.py`：データの読み込み
- 名前：espmk1（重みは `exp/encac_espmk1_20261116/best.pt`，実時間用は exp-realtime-epd の `weights/encac_espmk1.pt`）

## 手順

### 0. 準備

- [03 音声認識](../03_stage1_asr/README.md) の重み：`exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth`
- 学習データ `data/train_gxac`：[01 データの準備](../01_data/README.md) の 5
- 開発データ `data/dev_g`（4,000 区間で検証し，損失が最小の重みを残す）

### 1. 学習（GPU 1 枚，メモリ最大約 27 GB）

```bash
cd egs2/cejc/asr1/local/encac
export ENC=espnet:<asr1>/exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth
export ASR1=<asr1>                 # data/ のある場所（既定は /home/kobori/2025/espnet/egs2/cejc/asr1）
export PYTHONPATH=<espnet のリポジトリ>
MAXSEC=12 CTCW=1.0 python train_ac.py <asr1>/exp/encac_espmk1 --lr 5e-4 --train train_gxac --bs 8 --acc 8
```

| 指定 | 意味 |
| --- | --- |
| `CTCW=1.0` | 音声認識の補助タスク（その区間の書き起こしに対する CTC 損失）の重み．0 で補助タスクなし |
| `--train train_gxac` | 前の区間とつないだ音声も足す．`train_gx` で足さない |
| `MAXSEC=12` | 入力の長さの上限（秒） |
| `PRE=0.6` | 学習で読む位置を，区間末の 0.6 秒前からにする（既定は区間末の後 0〜2 秒だけ） |
| `--freeze_enc` | エンコーダを固定して線形層だけ学習する |

学習 375,612 区間で 1 エポック約 11,700 回の更新．2 エポック．

### 2. 実時間のプログラムに載せる

```bash
python ~/2026/work/exp-realtime-epd/tools/export_encac.py <asr1>/exp/encac_espmk1/best.pt \
    ~/2026/work/exp-realtime-epd/weights/encac_espmk1.pt
cd ~/2026/work/exp-realtime-epd && python tools/check_encac.py weights/encac_espmk1.pt --device cuda   # 学習側と同じ確率が出るかを確かめる
```

`ENC` と `PYTHONPATH` は学習と同じにしておく．

### 3. 評価

[07 実時間の評価](../07_realtime_eval/README.md) で，スコアを書き出し（`kobori_rt_dump.sh`），開発データで応答の決め方を選び（`rt_pick.py`），評価データで実際に流して確かめる（`kobori_rt_real.sh`）．
実際に回したジョブは `local/realtime_eval/kobori_chain17.sh`（学習〜書き出し）と `kobori_chain20.sh`（評価データでの確認）．

## アブレーション（工夫を 1 つずつ外す）

提案手法から 1 か所だけ変える．評価は提案手法と同じ（決め方は外した版ごとに開発データで選び直す）．

| 外す工夫 | 名前 | 変えるところ | ジョブ |
| --- | --- | --- | --- |
| 音声認識の補助タスク | espm | `CTCW=0`（指定しない） | 11/11 に手で流した（上の学習のコマンドから `CTCW=1.0` を除いたもの．スコアの書き出しは `kobori_chain1.sh`） |
| エンコーダの学習 | espmfz | `--freeze_enc`（固定すると補助タスクは効かないので `CTCW=0`） | `kobori_chain2.sh` |
| 前の区間とつないだ音声 | espk1 | `--train train_gx` | `kobori_chain19.sh` |
| 区間末より後だけで学習 | espmk1pre | `PRE=0.6` | `kobori_chain19.sh` |

## エンコーダを取り替える比較（参考）

`ENC` を変えると，同じ仕組みで別のエンコーダを試せる（`run_enc.sh`・`run_enc2.sh`）．

| `ENC` | エンコーダ |
| --- | --- |
| `espnet:<pth>` | 03 の contextual block conformer |
| `whisper:<名前>[:chunk=N]` | Whisper（small・medium） |
| `w2v:<名前>` | wav2vec 2.0（reazon-research/japanese-wav2vec2-large-rs35kh） |
| `mimi:<名前>` | Mimi |

whisper・w2v・mimi は conda 環境 `easyturn`＋transformers 4.46.3 で動かす．

## 注意

- 実時間のプログラム側の `epd/models/encac.py`・`tools/export_encac.py`・`tools/check_encac.py` は，exp-realtime-epd にまだコミットしていない（silver13 の作業ツリーにだけある）．
- 補助タスクの追加などは `local/realtime_eval/patch_*.py` で `train_ac.py` を書き換えて足した．今の `train_ac.py` には反映済みなので，流し直す必要はない．
