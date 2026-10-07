# 過去発話の入れ方 — テキスト版 / 全フレーム版 の実装説明

発話区間末検出 Stage2 に追加した2つのバリエーションの「何を・どうやっているか」の説明。
これまでの pooling 版（過去1発話→1ベクトル）に対する代替。

- 対象モデル: `espnet2/asr/espnet_model_turntaking_xfmr.py`（`model: turntaking_xfmr`）
- 共通の前提: **エンコーダは凍結**（Stage1 の ASR をそのまま使う、LoRA/Adapter なし）。学習するのは Self-Attention ヘッド（＋テキスト版は射影層）のみ。ASR の CER は Stage1 と同一。


> **⚠ Stage1 の前提について（2026-09-24 追記）**
> 本書の CER および Stage2 の全数値は、Stage1 = `20260713-pureasr`（**CER 23.3**、
> `block_size: 18`）を前提としている。その後 Stage1 を改善し
> `20260921-pureasr-blk42-sp`（**CER 20.7**、`block_size: 42`）を得た。
> **新しい Stage1 に乗り換える場合、`block_size` が変わるため本書の実験はすべて
> やり直しになる**（本書の数値と直接は比較できない）。→ `docs/stage1_asr_improvement.md`

---

## 0. おさらい：Self-Attention ヘッドの入り口

Stage2 のヘッドは、**「過去の表現」と「現発話フレーム列」を1本に並べて Self-Attention** し、
**現発話の最終フレーム位置の出力**を MLP に通してタグを出す。

```
入力列 = [ 過去の表現 (Np本) ; 現発話フレーム c1..cT ]      （長さ Np+T）
          └───────┬────────┘   └────────┬────────┘
     過去をどう表すか（★ここが今回の違い）   現発話（全手法共通・pooling しない）
                       │
                 Self-Attention（1層）
                       │
        現発話の最終フレーム位置(c_T)の出力 → MLP → 継続/完了/相槌
```

- 各トークンには **セグメント埋め込み**（過去=0 / 現発話=1）と **位置符号（sinusoidal）** を加算。
- 「過去の表現」の作り方だけを差し替えれば、下流（Self-Attention → 読み出し → MLP）は全手法で共通。

**手法ごとの「過去の表現」の違い（＝入力列の左側 Np 本が何か）**

| 手法 | 過去の表現（Np本の中身） | Np |
|---|---|---|
| pooling版 max/attn | 過去N発話を **1本のベクトルに要約** | 1 |
| 事前計算 past_vec | 過去発話ごとの max-pool ベクトル | N |
| **全フレーム版（今回②）** | 過去音声の **全フレーム**（要約しない） | 過去の総フレーム数 |
| **テキスト版（今回①）** | 過去テキストの **トークン埋め込み列** | 過去の総トークン数 |

---

## 1. テキスト版（過去はテキスト・音声は現発話のみ）

### 狙い
- ラベル付けは LLM が**テキストだけ**で高精度（F1 0.805）だった。過去は**テキストで足りる**かを検証。
- 音声を pooling して潰す代わりに、**語彙・文末表現・フィラー・言い直しをそのまま**渡す。

### データの作り方: `local/build_past_text.py`
各発話について、直前N発話（same=同一話者 / session=同一セッション）の**書き起こしを古→新の順に連結**し、
`tokens.txt` で**トークンID列**に変換して出力する。

```
（例）現発話の直前2発話が「うん」「そうだよね」だった場合
   → トークン列 [うん, そう, だ, よ, ね] → ID列 "12 5 8 …"
   出力 past_text.scp:  <uttid> 12 5 8 …
```
- **フィラー `(F あの)`・言い直し `(D デンワ)` のタグはそのまま含める**（意図的に残す。発話が続くかの手がかり）。
- 過去なし（会話先頭）は ID `0`（=blank）1個 → 学習側で「過去なし」として扱う。
- データ型は `text_int`（asr.sh で配線）。

### モデルの処理（`espnet_model_turntaking_xfmr.py`）
```
past_text (トークンID列)
      │  decoder.embed         ← Stage1 の Transducer デコーダの埋め込み（2611語→512次元・凍結）
      ▼
   埋め込み列 (L, 512)
      │  past_text_proj        ← 学習可能な線形層（512→256）★ここだけ新規学習
      ▼
   過去の表現 (L, 256) ─────────► Self-Attention ヘッドの左側 Np=L 本として入力
```
- **埋め込みは凍結再利用**（ASR が学習した語彙表現をそのまま使う・パラメータ増を抑える）。
- 学習で増えるのは **射影層 `past_text_proj`（512×256）** だけ。

### 設定・実行
- config: `myconf/train_asr_turntaking_xfmr_text.yaml`（`model_conf: tt_past_text: true`）
- 学習: `scope=same NS="1 2 3 4 5" pureasr_tag=20260713-pureasr ./run_exp_xfmr_text_series.sh`
- 評価: `scope=same NS="1 2 3 4 5" ./eval_xfmr_text_series.sh`

### 注意
学習・評価とも **正解書き起こし(reference)** を過去テキストに使う。実運用では過去も ASR 出力になるため、
本実験は「**テキストが効くかの上限**」を見るもの、と位置づける。

---

## 2. 全フレーム版（pooling しない ＝ Transformer-XL 的）

### 狙い
- pooling は過去1発話を1本に**強く圧縮**する。この圧縮が情報を落としている可能性。
- **圧縮せず過去の全フレームを残し**、現発話がそれに直接 attention する。
- これは **Transformer-XL のメモリ機構**（前セグメントの隠れ状態を要約せずメモリに保持し、現セグメントが attention する）に対応する考え方。

### Transformer-XL との対応・違い
| Transformer-XL | 本実装（全フレーム版） |
|---|---|
| 前セグメントの隠れ状態を**メモリ**として保持 | 過去発話の**全フレーム**を入力列の左側に置く |
| メモリには勾配を流さない | 過去フレームは**凍結エンコーダ出力**（勾配なし・`no_grad`） |
| 相対位置符号 | **未実装**（絶対位置符号 sinusoidal のまま）＝簡易版 |

→ 本実装は Transformer-XL の**核心（要約せずメモリに attention）だけを取り入れた簡易版**。
相対位置符号は入れていないので、「全フレームを残すと効くか」が確認できたら本格版に進む余地がある。

### データ
- pooling版と同じ **過去音声 `past_speech`**（過去N発話を連結した音声, `concat_sound`）を使う。
- 追加のストレージは不要（音声のパスを持つだけ）。

### モデルの処理
```
past_speech (過去N発話の連結音声)
      │  凍結エンコーダ (no_grad)          ← Stage1 のエンコーダ・勾配も統計も更新しない
      ▼
   過去の全フレーム (Lp, 256)   ← ★ pooling しない（max/attn 版はここで1本に潰す）
      └──────────────► Self-Attention ヘッドの左側 Np=Lp 本として入力
```
- max/attn pooling 版とは、この「全フレームを潰すか否か」だけが違う（同じ `past_speech` 経路）。
- `tt_past_pool` の値で分岐:
  - `max` … フレーム最大値で1本に
  - `attn` … 学習クエリで加重平均して1本に（`PastAttnPool`）
  - `frames` … **潰さず全フレーム**（本手法）

### 設定・実行
- config: `myconf/train_asr_turntaking_xfmr_pool_frames.yaml`（`tt_past_pool: frames`）
- 学習: `pool=frames scope=same NS="1 2 3 4 5" pureasr_tag=20260713-pureasr ./run_exp_xfmr_pool_series.sh`
- 評価: `pool=frames scope=same NS="1 2 3 4 5" ./eval_xfmr_pool_series.sh`

### 注意
- 過去の全フレームを入れると**入力列が長くなる**（Np = 過去の総フレーム数）。
  N が大きい・session だと系列が伸び、**メモリ・計算が増える**。まず **N=1,2 same** から。
- 学習中は過去音声も毎ステップ符号化する（凍結・no_grad）ため、pooling 版よりやや遅い。

---

## 3. 3手法の比較（過去の「圧縮度」の観点）

```
強く圧縮 ←──────────────────────────────────────→ 圧縮しない
  max / attn pooling        テキスト（トークン列）        全フレーム
  過去 → 1ベクトル            過去 → トークン埋め込み列       過去 → 全フレーム
  （情報を潰す）              （語彙・文末表現を保持）        （音響も全部保持）
```

- **テキスト版**: 音響（韻律など）は捨てるが、語彙・文末・フィラーは完全に保持。
- **全フレーム版**: 音響も含め何も捨てないが、系列が長く重い。
- どちらも「過去を賢く残せば効くか」を、**pooling 版と直接比較**して検証する。

---

## 4. 関連ファイル

| 種別 | パス |
|---|---|
| モデル | `espnet2/asr/espnet_model_turntaking_xfmr.py`（`tt_past_text`, `tt_past_pool: frames`, `past_text_proj`, `PastAttnPool`） |
| データ生成 | `local/build_past_text.py`（テキスト）／`local/build_past_audio.py`（音声・既存） |
| config | `myconf/train_asr_turntaking_xfmr_text.yaml`, `..._pool_frames.yaml` |
| 学習 | `run_xfmr_text.sh` / `run_exp_xfmr_text_series.sh`（テキスト）、`run_xfmr_pool.sh` / `run_exp_xfmr_pool_series.sh`（frames は `pool=frames`） |
| 評価 | `eval_xfmr_text_series.sh`、`eval_xfmr_pool_series.sh`、`local/turntaking/evaluate_multitask.py`（`--past_text` / `--past_pool`） |
