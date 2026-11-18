# espnet_exp

CEJC（日本語日常会話コーパス）を用いた，発話区間末の判定（継続 / 完了 / 相槌）の ESPnet2 レシピと，そのための ESPnet の改変・追加モジュール．
話者が黙ったときに「伝えたいことを言い終えたか」を意味から判定し，音声対話システムが遮らず，待たせずに応答できるようにする．

## やりたいことから探す

| やりたいこと | 読む README |
| --- | --- |
| CEJC を学習・開発・評価に分けて，学習データを作る | [01 データの準備](egs2/cejc/asr1/docs/experiments/01_data/README.md) |
| 区間末のラベル（継続・完了・相槌）を LLM で付ける・付け直す | [02 ラベル](egs2/cejc/asr1/docs/experiments/02_labels/README.md) |
| 音声認識（Stage1）を学習する．提案手法のエンコーダの初期値になる | [03 音声認識](egs2/cejc/asr1/docs/experiments/03_stage1_asr/README.md) |
| 比べる相手の「エンコーダ固定＋Transformer」（ESPnet の Stage2）を学習する | [04 エンコーダ固定＋Transformer](egs2/cejc/asr1/docs/experiments/04_xfmr/README.md) |
| **提案手法**（エンコーダも学習＋音声認識を補助タスクとしたマルチタスク学習）を学習する．アブレーションも | [05 提案手法](egs2/cejc/asr1/docs/experiments/05_proposed/README.md) |
| 比べる相手の Easy Turn・Smart Turn・VAP・silero-vad を用意する | [06 比べる相手](egs2/cejc/asr1/docs/experiments/06_baselines/README.md) |
| 実時間で評価する（応答の決め方を選ぶ・cut-in rate と latency を出す） | [07 実時間の評価](egs2/cejc/asr1/docs/experiments/07_realtime_eval/README.md) |
| 別の対話データ（waseda_soma・Hazumi）で確かめる | [08 別の対話データ](egs2/cejc/asr1/docs/experiments/08_other_corpora/README.md) |

## 全体の流れ

```
01 データの準備 ─→ 02 ラベル ─→ 03 音声認識（Stage1）─┬→ 04 エンコーダ固定＋Transformer ─┐
                                                       └→ 05 提案手法 ───────────────────┼→ 07 実時間の評価 ─→ 08 別の対話データ
                                       06 比べる相手（Easy Turn など）──────────────────┘
```

今の報告で使っているのは，05 の提案手法を 07 で評価した結果と，04・06 の比べる相手を同じく 07 で評価した結果．

## 動かす場所と環境

- 作業ディレクトリは `egs2/cejc/asr1`．研究室のサーバでは `~/2025/espnet/egs2/cejc/asr1` にある（silver11・silver13 にそれぞれ別のコピーがある）．
- `tt_batch_logs/*.sh` と `local/encac`・`local/realtime_eval` のスクリプトには，研究室のサーバのパス（`/home/kobori/...`・`/autofs/...`）が書いてある．別の場所で動かすときは書き換える．
- conda 環境
  - `espnet`：ESPnet の学習・評価，提案手法の学習
  - `easyturn`：Easy Turn
  - `epd`：実時間の評価（別リポジトリ exp-realtime-epd）
  - `llm`：vLLM によるラベル付け
- 実時間の評価とデモは，別リポジトリ **exp-realtime-epd**（非公開）の `epd.engine` を使う．

## このリポジトリに入れないもの

CEJC・waseda_soma・Hazumi の音声と書き起こし，学習した重み，LLM の判定結果は入れない（`.gitignore` で `data/`・`dump/`・`exp/`・`tt_analysis/` などを外している）．これらは研究室のサーバ（`/autofs/diamond5/share/users/kobori/` など）にだけ置く．

## これより前の実験

Stage2 の過去の文脈・BERT との融合・早期確定などの記録は `egs2/cejc/asr1/docs/`（`HANDOFF.md`・`plan.md` など）にある．
