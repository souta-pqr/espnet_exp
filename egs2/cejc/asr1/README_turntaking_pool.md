# 発話区間末予測（Stage2）：過去発話の要約手法の比較

現発話の区間末（継続 / 完了 / 相槌）を予測する Stage2 モデルで、**過去 N 発話をどんな長さの
記憶スロット列にしてヘッドへ渡すか**を比較する実験セット。

Stage1（純粋 ASR）のエンコーダを凍結し、Stage2 では self-attention 1 層 ＋ MLP ＋ 過去要約
モジュールだけを学習する。凍結部は `train()` でも eval 固定なので **CER は Stage1 と数値的に同一**。
比較指標は区間末タグの macro-F1 とクラス別 F1 のみ。

実装: `espnet2/asr/espnet_model_turntaking_xfmr.py`

## 手法一覧（`tt_past_pool`）

| pool | 分類 | 出力スロット数 Np | つまみ | 出典 |
| --- | --- | --- | --- | --- |
| `conv` | 固定率・soft | ⌈Tp / c⌉ | `tt_compress_ratio` (c) | Rae+ ICLR2020, arXiv:1911.05507 |
| `query` | 固定長・soft | M | `tt_compress_slots` (M) | Lee+ ICML2019 arXiv:1810.00825 / Jaegle+ ICML2021 arXiv:2103.03206 |
| `topk` | 固定長・hard | min(M, Tp) | `tt_compress_slots` (M) | Rae+ ICLR2020 "most-used", arXiv:1911.05507 |
| `utt` | 会話単位 | 有効発話数 × k | `tt_utt_slots` (k) | 出典なし（対照条件。境界推定の一般解は CIF: Dong+ ICASSP2020 arXiv:1905.11235） |
| `attn` / `max` | 既存・固定長 | 1 | — | 比較の下限（`query` の M=1 に相当） |
| `frames` | 圧縮なし | Tp | — | 比較の上限（`conv` の c=1 に相当） |

端点が既存手法と一致するので、掃引の両端がそのまま既存手法との比較になる。

## 前提

1. Stage1（純粋 ASR）が学習済みで、`exp/asr_<pureasr_tag>/valid.loss.ave.pth` があること
   ```
   ./run_pureasr.sh          # asr_tag=<日付>-pureasr
   ```
2. `asr.sh` の **Stage 11（ASR Training）に past_* の配線があること**。
   これが無いと `past_speech` が学習ジョブに渡らず、`tt_past_pool` を何にしても
   「現発話のみ」のモデルが学習される（過去に一度この状態で実験を回してしまった）。
   各 `run_pool_*.sh` は起動時にこの配線を検査し、無ければ学習せず中止する。
   ```
   awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh
   ```
3. 過去音声（`past_speech.scp` と `past_bounds.scp`）は `local/build_past_audio.py` が生成する。
   `run_xfmr_pool.sh` が無ければ自動で作るので、通常は明示的に呼ばなくてよい。

## 過去発話の入れ方

- 単位は **segments 1 行 ＝ 1 発話セグメント**をそのまま（固定長の窓ではない）
- `scope=same`（既定）は同じ話者チャネル、`scope=session` は `_IC\d+` を落とした同一セッション全話者
- 直前 N 発話を古→新の順に連結。過去発話の start が「現発話 start − `max_past` 秒」より前になった時点で打ち切り
- 連結は音声ファイルの結合のみで、**発話間のポーズは含まれない**
- 過去がない発話（会話先頭）は 20 ms の無音（train_nodup で約 1.5%）

CEJC train_nodup / `scope=same` での実測:

| N | 過去長 中央値 | 平均 | 90%点 | 平均 Tp（フレーム） |
| --- | --- | --- | --- | --- |
| 1 | 1.03 s | 1.45 s | 3.14 s | 約 44 |
| 5 | 6.03 s | 6.74 s | 12.10 s | 約 204 |

フレーム長 = hop 132 サンプル × conv2d 1/4 サブサンプリング = 528 サンプル（≒ 33 ms）。
N=1 では平均 44 フレームしかなく、M=8 スロットでは圧縮率がほとんど稼げないため、
**手法差を見るなら N=5 を主軸にする**のがよい。

## 実行

### 手法ごとに 1 本ずつ

```bash
pureasr_tag=20260713-pureasr N=5 ./run_pool_conv.sh            # c は平均Tpから自動決定（N=5 → 26）
pureasr_tag=20260713-pureasr N=5 slots=8 ./run_pool_query.sh
pureasr_tag=20260713-pureasr N=5 slots=8 ./run_pool_topk.sh
pureasr_tag=20260713-pureasr N=5 ./run_pool_utt.sh             # k=1
```

### つまみを振る掃引（N 固定）

出典のある 3 手法のみ受け付ける。`VALS` の意味は pool で変わる（conv=c / query,topk=M）。

```bash
pool=conv  VALS="1 8 16 32"  N=5 pureasr_tag=... ./run_exp_xfmr_pool_compress_sweep.sh
pool=query VALS="1 2 4 8 16" N=5 pureasr_tag=... ./run_exp_xfmr_pool_compress_sweep.sh
pool=topk  VALS="4 8 16"     N=5 pureasr_tag=... ./run_exp_xfmr_pool_compress_sweep.sh
pool=conv  VALS="1 8 16 32"  N=5 ./eval_xfmr_pool_compress_sweep.sh
```

### 過去発話数 N を振る掃引（つまみ固定）

```bash
pool=query slots=8 NS="1 2 3 4 5" scope=same pureasr_tag=... ./run_exp_xfmr_pool_series.sh
pool=query slots=8 NS="1 2 3 4 5" scope=same ./eval_xfmr_pool_series.sh
```

### スロット予算を揃えた手法間比較（「どれが有効か」）

conv の圧縮率を `past_bounds` の平均過去長から逆算し、全手法の平均スロット数を揃える。
学習済みタグは自動スキップ（`force=1` で再学習）。

```bash
budget=8 N=5 scope=same pureasr_tag=... ./run_exp_xfmr_pool_compare.sh
budget=8 N=5 scope=same ./eval_xfmr_pool_compare.sh     # macro-F1 降順の 1 枚の表
```

### 単発の評価

```bash
pool=query slots=8 scope=same N=5 ./eval_xfmr_pool.sh
```

## 実験タグ

`exp/asr_<日付>-turntaking-xfmr-pool_<pooltag>-<scope>_n<N>_p<max_past>` に設定値が入る
（`pool_conv_c26-same_n5_p30`、`pool_query_m8-same_n5_p30` など）。
圧縮率やスロット数を振っても実験が上書きされない。

## 確認のしかた

学習を起動したら、過去が本当に入っているかを最初に確認する。

```bash
grep -c past_speech exp/asr_*-pool_query_m8-same_n5_p30/config.yaml   # 0 なら過去が入っていない
```

評価表の CER 列は全手法で同一になるはず（凍結が壊れていれば違う値が出る）。

## ファイル

| ファイル | 役割 |
| --- | --- |
| `run_pureasr.sh` | Stage1（純粋 ASR）。凍結エンコーダを作る |
| `run_xfmr_pool.sh` | Stage2 の共通ドライバ（データ生成 → config 生成 → 学習 → デコード） |
| `run_pool_{conv,query,topk,utt}.sh` | 手法ごとの学習スクリプト。出典・つまみ・配線検査つき |
| `run_exp_xfmr_pool_compress_sweep.sh` / `eval_…` | つまみの掃引と集計 |
| `run_exp_xfmr_pool_series.sh` / `eval_…` | N の掃引と集計 |
| `run_exp_xfmr_pool_compare.sh` / `eval_…` | 予算を揃えた手法間比較と集計 |
| `eval_xfmr_pool.sh` | 単発評価（`local/turntaking/evaluate_multitask.py` を呼ぶ） |
| `local/build_past_audio.py` | `past_speech.scp` / `past_bounds.scp` の生成 |
| `myconf/train_asr_turntaking_xfmr_pool_*.yaml` | 手法ごとの学習 config |
