# 発話区間末予測（Stage2）：過去発話の入れ方の比較

現発話の区間末（継続 / 完了 / 相槌）を予測する Stage2 モデルで、**過去 N 発話をどう要約して
ヘッドへ渡すか**、および**エンコーダを部分適応するか**を比較する実験セット。

Stage1（純粋 ASR）のエンコーダを読み込み、Stage2 では self-attention 1 層 ＋ MLP ＋ 過去要約
モジュールを学習する。完全凍結版は CER が Stage1 と数値的に同一。PEFT 版（LoRA / Adapter）は
エンコーダが適応するため CER が変わるので、タグ性能と CER 劣化の両方を見る。

実装: `espnet2/asr/espnet_model_turntaking_xfmr.py`

## 最重要の前提：Stage 11 の past_* 配線

`asr.sh` の past_* データ配線は以前 **Stage 10（collect stats）にしか書かれておらず、
Stage 11（ASR Training）に無かった**。そのため過去発話が学習ジョブに渡らず、
`tt_past_pool` を何にしても「現発話のみ」のモデルが学習されていた
（異なる手法・異なる N の run が valid tag_acc までビット単位で一致する、という形で発覚）。

**この配線が入っていることを必ず確認する。** 各 `run_pool_*.sh` / `run_peft_*.sh` は
起動時に検査し、無ければ学習せず中止する。

```bash
awk '/Stage 11: ASR Training/{s=1} s && /past_speech\.scp,past_speech,concat_sound/{f=1} END{exit !f}' asr.sh
```

学習を起動したら、最初のエポックが回る前に確認する。

```bash
grep -c past_speech exp/asr_*-pool_attn-same_n5_p30/config.yaml   # 0 なら過去が入っていない
```

## 手法一覧

### 過去要約（`tt_past_pool`）

| pool | 分類 | 出力スロット数 Np | つまみ | 出典 |
| --- | --- | --- | --- | --- |
| `max` | 固定長・非学習 | 1 | — | 比較の下限（`conv` の c≥Tp と数値一致） |
| `attn` | 固定長・soft | 1 | — | `query` の M=1・単一ヘッドと同機構 |
| `frames` | 圧縮なし | Tp | — | 比較の上限（`conv` の c=1 と一致） |
| `conv` | 固定率・soft | ⌈Tp / c⌉ | `tt_compress_ratio` (c) | Rae+ ICLR2020, arXiv:1911.05507 |
| `query` | 固定長・soft | M | `tt_compress_slots` (M) | Lee+ ICML2019 arXiv:1810.00825 / Jaegle+ ICML2021 arXiv:2103.03206 |
| `topk` | 固定長・hard | min(M, Tp) | `tt_compress_slots` (M) | Rae+ ICLR2020 "most-used", arXiv:1911.05507 |
| `utt` | 会話単位 | 有効発話数 × k | `tt_utt_slots` (k) | 出典なし（対照条件。境界推定の一般解は CIF: Dong+ ICASSP2020 arXiv:1905.11235） |

`max` / `attn` / `frames` の config には、圧縮系と条件をそろえるため `tt_rel_pe: true` と
`tt_past_specaug: false` を追加してある（以前は入っていなかった）。

### エンコーダ適応（`tt_adapt`）

過去要約とは直交する軸。既定は `none`（完全凍結）。

| adapt | 内容 | 設定 | 出典 |
| --- | --- | --- | --- |
| `lora` | 自己注意 Q,V に低ランク行列を挿入 | rank=8, alpha=8, dropout=0.05 | Hu+ ICLR2022, arXiv:2106.09685 |
| `adapter` | 各ブロックの主 FFN 出力に Bottleneck Adapter を加算 | bottleneck=32, 12 箇所, max_ratio=0.4 | Houlsby+ ICML2019, arXiv:1902.00751 |

## 過去発話の入れ方

- 単位は **segments 1 行 ＝ 1 発話セグメント**そのまま（固定長の窓ではない）
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
N=1 では平均 44 フレームしかなく M=8 スロットでは圧縮率が稼げないため、
**手法差を見るなら N=5 を主軸にする**。

## 実行

前提: Stage1 が学習済みで `exp/asr_<pureasr_tag>/valid.loss.ave.pth` があること（`./run_pureasr.sh`）。

### 既存手法の再実験（過去が入っていなかったぶん）

```bash
pureasr_tag=20260713-pureasr N=5 ./run_pool_max.sh        # max pooling
pureasr_tag=20260713-pureasr N=5 ./run_pool_attn.sh       # attention pooling
pureasr_tag=20260713-pureasr N=5 ./run_peft_lora.sh       # LoRA（過去要約は既定 attn）
pureasr_tag=20260713-pureasr N=5 ./run_peft_adapter.sh    # Adapter（同上）
```

PEFT 版は `pool=` で過去要約を変えられる（例 `pool=query slots=8 ./run_peft_lora.sh`）。
タグは `pool_attn-lora-same_n5_p30` のように `-lora` / `-adapter` が付く。

### 新規手法

```bash
pureasr_tag=20260713-pureasr N=5 ./run_pool_conv.sh            # c は平均Tpから自動決定（N=5 → 26）
pureasr_tag=20260713-pureasr N=5 slots=8 ./run_pool_query.sh
pureasr_tag=20260713-pureasr N=5 slots=8 ./run_pool_topk.sh
pureasr_tag=20260713-pureasr N=5 ./run_pool_utt.sh             # k=1
```

### 過去なしの床（N=0）

```bash
pureasr_tag=20260713-pureasr ./run_xfmr_n0.sh    # past_* を一切渡さない（真の Np=0）
./eval_xfmr_n0.sh
```

### 掃引と比較

```bash
# つまみを振る（出典のある 3 手法のみ。VALS は conv=c / query,topk=M）
pool=conv  VALS="1 8 16 32"  N=5 pureasr_tag=... ./run_exp_xfmr_pool_compress_sweep.sh
pool=conv  VALS="1 8 16 32"  N=5 ./eval_xfmr_pool_compress_sweep.sh

# 過去発話数 N を振る（つまみ固定）
pool=query slots=8 NS="1 2 3 4 5" pureasr_tag=... ./run_exp_xfmr_pool_series.sh
pool=query slots=8 NS="1 2 3 4 5" ./eval_xfmr_pool_series.sh

# スロット予算を揃えた手法間比較（conv の c を平均Tpから逆算してそろえる）
budget=8 N=5 pureasr_tag=... ./run_exp_xfmr_pool_compare.sh
budget=8 N=5 ./eval_xfmr_pool_compare.sh          # macro-F1 降順の 1 枚の表
```

### 単発の評価

```bash
pool=attn N=5 ./eval_xfmr_pool.sh                          # 完全凍結版
pool=attn N=5 tag_suffix=-lora ./eval_xfmr_pool.sh         # LoRA 版
pool=attn N=5 tag_suffix=-adapter ./eval_xfmr_pool.sh      # Adapter 版
```

## 実験タグ

```
exp/asr_<日付>-turntaking-xfmr-pool_<pooltag><tag_suffix>-<scope>_n<N>_p<max_past>
```

`pool_conv_c26-same_n5_p30`、`pool_query_m8-same_n5_p30`、`pool_attn-lora-same_n5_p30` のように
設定値が入るので、つまみや PEFT を振っても実験が上書きされない。
`stats` ディレクトリは過去データと形状だけで決まるので、同じ pool の PEFT 版と共有する。

CER 列は完全凍結版なら全手法で同一になるはず（違えば凍結が壊れている）。PEFT 版は変わるのが正常。

## ファイル

| ファイル | 役割 |
| --- | --- |
| `run_pureasr.sh` | Stage1（純粋 ASR）。凍結エンコーダを作る |
| `run_xfmr_pool.sh` | Stage2 の共通ドライバ。`base_cfg` / `tag_suffix` で PEFT 版に流用できる |
| `run_pool_{max,attn,conv,query,topk,utt}.sh` | 過去要約ごとの学習スクリプト |
| `run_peft_{lora,adapter}.sh` | エンコーダ部分適応版 |
| `run_xfmr_n0.sh` / `eval_xfmr_n0.sh` | 過去なし（N=0）の床 |
| `run_exp_xfmr_pool_compress_sweep.sh` / `eval_…` | つまみの掃引と集計 |
| `run_exp_xfmr_pool_series.sh` / `eval_…` | N の掃引と集計 |
| `run_exp_xfmr_pool_compare.sh` / `eval_…` | 予算を揃えた手法間比較と集計 |
| `eval_xfmr_pool.sh` | 単発評価（`local/turntaking/evaluate_multitask.py` を呼ぶ） |
| `local/build_past_audio.py` | `past_speech.scp` / `past_bounds.scp` の生成 |
| `myconf/train_asr_turntaking_xfmr_pool_*.yaml` | 過去要約ごとの config |
| `myconf/train_asr_turntaking_xfmr_peft_*.yaml` | PEFT 版の config（過去要約は driver が `pool=` に合わせて書き換える） |
