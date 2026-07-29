# espnet_multitask

CEJC を用いた発話区間末タグ推定（継続 / 完了 / 相槌）の ESPnet2 レシピと、そのための ESPnet 改変・追加モジュール。

**これは ESPnet 本体のオーバーレイです。**単体では動きません。ESPnet を clone した上に、
同じディレクトリ構造のままファイルを重ねて使います。

```bash
git clone https://github.com/espnet/espnet.git
cd espnet && (tools のセットアップ)
rsync -a /path/to/espnet_multitask/espnet2/  espnet2/
rsync -a /path/to/espnet_multitask/egs2/     egs2/
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
| `espnet2/` | 改変ファイルと自作モデル（下記） |

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

`utils/`, `scripts/`, `pyscripts/` の一部は ESPnet ルートの `utils/` を指すシンボリックリンクです
（例: `pyscripts/audio/trim_silence.py -> ../../../../../utils/trim_silence.py`）。
このリポジトリ単体ではリンク切れに見えますが、**ESPnet clone の上に重ねた時点で解決します**。
上流のファイルを二重に持たないための構造なので、そのままにしてあります。

例外として `utils/simple_dict.sh` はコピー元の時点でリンク切れでした（本リポジトリ由来の問題ではありません）。

## 含めていないもの

- `data/`, `dump/`, `exp/` — 音声・特徴量・学習済みモデル
- `local/data.sh`, `local/score.sh` — 元のディレクトリにも存在しません。
  データ準備は `cejc_data/` 側の前処理コードで行われており、本リポジトリには含めていません。
- 参照されていない config、分析・診断スクリプト、設計メモ
