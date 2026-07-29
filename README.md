# espnet_multitask

CEJC を用いた発話区間末タグ推定（継続 / 完了 / 相槌）の ESPnet2 レシピと、そのための ESPnet 改変・追加モジュール。

## セットアップ

Python パッケージ（`espnet2/`, `espnet/`）とルート `utils/` を同梱しているので、
このリポジトリ単体で `run_*.sh` が動きます。必要なのは以下だけです。

```bash
# 1. Python 環境を用意し、このリポジトリを editable install する
#    （既に espnet が editable install されている環境なら、張り替えになる）
cd /path/to/espnet_multitask
pip install -e .

# 2. tools/activate_python.sh を自分の環境に合わせる
#    （path.sh がこれを source する。無いと素の python3 が使われて
#      ModuleNotFoundError: No module named 'espnet2' で即死する）
vi tools/activate_python.sh

# 3. データを繋ぐ（音声はリポジトリに含めていない）
cd egs2/cejc/asr1
ln -s /path/to/original/data data
ln -s /path/to/original/dump dump
```

> **`pip install -e .` が必要な理由**: conda 環境に別の espnet が editable install されていると、
> PEP 660 の meta-path finder が働くため `PYTHONPATH` では上書きできず、`import espnet2` が
> 元のディレクトリを指し続けます。専用の環境を作るか、このリポジトリで install を張り替えてください。

### exp/ の扱い（注意）

`exp/` は**元の作業ディレクトリと共有しないでください**。同じ実験タグの学習が同時に走ると、
同一ディレクトリに 2 プロセスが書き込んで壊れます。Stage1 の学習済みモデルだけを
読み取り用に繋ぐのが安全です。

```bash
mkdir -p exp
ln -s /path/to/original/exp/asr_<Stage1タグ> exp/asr_<Stage1タグ>
```

## 構成

| パス | 中身 |
| --- | --- |
| `egs2/cejc/asr1/asr.sh` | パイプライン本体（ESPnet 標準から改変） |
| `egs2/cejc/asr1/run_*.sh` | 各実験の起動スクリプト |
| `egs2/cejc/asr1/eval_*.sh` | タグ性能の評価と表出力 |
| `egs2/cejc/asr1/myconf/` | 学習・デコード config（参照されているもののみ） |
| `egs2/cejc/asr1/local/` | 過去文脈データの生成、タグ評価 |
| `egs2/cejc/asr1/{utils,scripts,pyscripts}/` | ESPnet 標準の補助スクリプト |
| `espnet2/`, `espnet/` | ESPnet 本体（改変ファイルと自作モデルを含む・下記） |
| `utils/` | ESPnet ルートの共通スクリプト（レシピ内 symlink の解決先） |
| `tools/` | `extra_path.sh` と `activate_python.sh` |

## espnet2 側の内訳

**自作モデル・タスク**

- `asr/espnet_model_turntaking_xfmr.py` — 凍結エンコーダ + self-attention head（現行の主系統）
- `asr/espnet_model_turntaking.py` — Conformer マルチタスク版（旧系統）
- `asr/espnet_model_cif_tag.py`, `asr/espnet_model_cif_teaser.py` — CIF ベース
- `asr/multitask_rnnt_model.py`, `asr/multitask_transformer_model.py`
- `asr/decoder/multitask_transducer.py`, `asr/layers/multitask_joint_network.py`
- `bin/turntaking_asr_{train,inference}.py`, `bin/cif_asr_{train,inference}.py`

**改変ファイル**

- `tasks/asr.py` — 上記モデルの登録、`tag_label` / `past_speech` / `past_vec` / `past_text` / `past_bounds` の入力定義
- `train/dataset.py` — `concat_sound` ローダ（過去N発話の音声を時間連結して読む）
- `train/trainer.py`, `asr/espnet_model.py`, `asr/transducer/*`, `asr/decoder/*`

`tasks/asr.py` は自作モデルを全て先頭で import するため、旧系統のモデルファイルも
揃っていないと import が通りません。

## 主系統の流れ

```bash
cd egs2/cejc/asr1

# Stage1: 純粋 ASR（エンコーダを作る）
./run_pureasr.sh

# Stage2: エンコーダを凍結し、タグ head だけを学習
#   過去発話の与え方を変えて比較する
pool=query slots=8  scope=same NS="1 5" pureasr_tag=<Stage1タグ> ./run_exp_xfmr_pool_series.sh
scope=same NS="1 2 3 4 5"      pureasr_tag=<Stage1タグ> ./run_exp_xfmr_text_series.sh

# 評価（継続/完了/相槌の F1・macro-F1・二値分離・CER）
pool=query slots=8 scope=same NS="1 5" ./eval_xfmr_pool_series.sh
```

## utils/ 内のシンボリックリンクについて

レシピ内の `utils/`, `scripts/`, `pyscripts/` の一部は、ESPnet ルートの `utils/` を指す
シンボリックリンクです（上流ファイルを二重に持たないための ESPnet の構造）。
ルート `utils/` を同梱しているので**すべて解決します**。

例外として `utils/simple_dict.sh` だけはコピー元の時点でリンク切れでした
（本リポジトリ由来の問題ではありません）。

## 外部ツールについて

`tools/` には `extra_path.sh` と `activate_python.sh` だけを入れています。
sctk（採点用 sclite）、sentencepiece、warp-transducer などのビルド成果物は
リポジトリに含めていません。`--token_type word` の現行レシピでは使いませんが、
必要になったら本家 ESPnet の `tools/Makefile` で用意してください。

## 含めていないもの

- `data/`, `dump/`, `exp/` — 音声・特徴量・学習済みモデル
- `local/data.sh`, `local/score.sh` — 元のディレクトリにも存在しません。
  データ準備は `cejc_data/` 側の前処理コードで行われており、本リポジトリには含めていません。
- 参照されていない config、分析・診断スクリプト、設計メモ
