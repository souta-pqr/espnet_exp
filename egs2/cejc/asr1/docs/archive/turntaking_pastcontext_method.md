# 過去文脈を足した発話区間末予測（手法解説）

作成日: 2026-06-16（特徴ベクトル方式に改訂）
関連: [turntaking_multitask_method.md](turntaking_multitask_method.md)（本体モデル）/ [turntaking_slides_plan.md](turntaking_slides_plan.md)
コード: `local/extract_context_vectors.py` / `run_ctxvec.sh` / `myconf/train_asr_turntaking_conformer_ctxvec.yaml`

> 「会話の過去文脈を足すと精度が上がる」という知見を発話区間末予測に応用する。
> **未来（trailing）は足さない**。過去（同一話者の直前 N 発話）を**特徴ベクトルとして条件付け**する。
> 重要：**現発話の音声・ASR は一切変えない**（過去音声を再生・再転写しない）。

---

## 1. 目的
- 発話区間末（no/yes/other）の判定に、**直前の発話文脈**が効くかを検証する。
- 制約：未来音声は入れない。現発話の ASR は不変に保つ（過去は“条件”として渡すだけ）。

---

## 2. どうやって過去情報を入れるか（特徴ベクトル方式）★本方式

過去発話の**音声を直接つなげて聞かせるのではなく**、過去発話を**符号化した固定長ベクトル**を
発話区間末ヘッドに条件として注入する。

### 2-1. 文脈ベクトル ctx_vec の作り方（過去N発話を「数字のまとまり」1本にする）

やりたいのは、現発話を判定するときに「**直前に同じ人が何を話していたか**」を
**256個の数字（ベクトル）1本**にまとめて渡すこと。手順は2ステップ。

**ステップ1：1つの発話を「ベクトル1本」に要約する（uvec を作る）**
- 発話の音声を学習済み encoder に通す → フレーム（33msごと）の特徴が並んだ列（長さ T × 256次元）。
- その列を**時間方向に平均**する → **発話1つ ＝ 256次元のベクトル1本（uvec）**。
- つまり uvec は「その発話の中身を 256 個の数字で要約したもの」。

**ステップ2：現発話の文脈ベクトルを作る（ctx_vec）**
- 現発話の**直前N発話（同じ話者）の uvec を平均** → **ctx_vec（256次元）**。
- 直前が無ければ（会話の最初など）→ ゼロベクトル（＝文脈なし）。

```
直前の発話 ─encoder─► フレーム列(T×256) ─時間平均─► uvec(256) ┐
（N=2 ならもう1つ前も同様に uvec を作る）                       ├─平均─► ctx_vec(256)
                                                              ┘
```

**具体例（N=1）**
```
同じ話者の連続発話:
  直前 :「あ い い だ い じ ょ ぶ だ よ」
  現発話:「す い ま せ ん」 ← これの no/yes/other を判定したい

 uvec(直前)     = encoder(「あいいだいじょぶだよ」の音声) を全フレーム平均 → 256次元
 ctx_vec(現発話) = uvec(直前)        ← これを現発話のヘッドに渡す
 ※ 現発話の音声・ASRテキストは「すいません」のまま（一切変更しない）
```

- encoder は**学習済みモデルの固定 encoder**（事前に1回通すだけ・学習中は更新しない外部特徴）。
  実装は `local/extract_context_vectors.py`（既定の抽出器＝新アーキ exp2 の encoder）。
> 「平均で1本にまとめる」部分を「**類似発話を集めて代表ベクトルにする（クラスタリング）**」等に
> 差し替えても、後段（モデル・配線）は変えなくてよい。

### 2-2. モデルへの渡し方（ASR は不変）
- 現発話は、encoder出力を**発話全体で平均**して 1 本のベクトル（現発話の要約・256次元）にする。
- そこに **ctx_vec(256) を concat → MLP** で no/yes/other を **1区間につき1つ**予測。
- **ASR（transducer）は encoder出力をそのまま使う＝完全に不変**。ctx は区間末ヘッドにだけ入る。
- 現発話の入力音声は **tight `[start, end]`**（未来＝trailing なし）。

```
  現発話音声 [start, end]（未来なし）
        │
   共有エンコーダ ── encoder出力 ──► transducer デコーダ → テキスト（ASR・不変）
        │
        └─ 発話全体で平均（現発話の要約 256次元）
                  │  ⊕ ctx_vec(256)   ← 過去N発話の要約（§2-1）
                  ▼
                MLP → 発話区間末タグ（no/yes/other を1つ）
```

### 2-3. パラメータ
- `N`：足す直前発話数（1, 2）／`max_past`：過去さかのぼり上限秒（既定 10）。
- 直前が無い区間は ctx_vec=ゼロ → 実質「文脈なし」として扱う。

---

## 2.5. 過去文脈の3方式（`ctx_vec` の作り方だけ切替・モデル/配線は不変）

「どの過去発話を、どう1本のベクトルにまとめるか」だけが違う。`local/extract_context_vectors.py` の
`--context_scope` で切替。注入の仕方（§2-2）・モデル・ASR は全方式で共通・不変。

### (A) same：同一話者の直前N発話（recency・自分の履歴）
- 同じ reco（=同一話者チャンネル）の**直前N発話**の uvec を平均 → ctx_vec。
- 「自分が直前に何を話していたか（談話の流れ）」を渡す。**相手の発話は入らない**。
```
（同一話者 IC01 の時系列）… 発話i-2  発話i-1 │ 発話i(現)     ← i-1,i-2 の uvec 平均
```

### (B) session：両話者の直前N発話（会話履歴・相手も含む）
- 同じ**セッション**（reco から `_IC\d+` を除いた `T007_007` 等）の**全チャンネル**を **start 時刻で統合**し、
  現発話の**直前N発話（話者問わず）**の uvec を平均 → ctx_vec。
- 同期録音なので異なる話者が共通時間軸で交互に並ぶ（マージ済みデータは不要・時刻ソートで実現）。
- 「**相手の直前ターンを含む会話文脈**」を渡せる（same の弱点＝相手が見えない、を補う）。
```
（セッション T007_007・両話者を時刻順）
 … IC02「うんうん」 IC01「〜あって」 │ IC01「そうそうなの」(現)   ← 直前N（相手の発話も含む）
```

### (C) cluster：過去の類似発話 top_k を集めて平均（similarity・過去のみ）
- 同一セッションの中で、**現発話より前の発話だけ**を対象に、uvec のコサイン類似度**上位 top_k** を取り、
  その平均 → ctx_vec（＝「過去の似た発話の代表ベクトル」）。**現発話より後は一切使わない**。
- recency（直前）ではなく **類似性**で過去から集める点が (A)(B) と異なる。
```
（セッションの過去発話のうち、現発話に似たものを top_k 件）
 過去発話群 ──cos類似で上位k件を選抜──► 平均 → ctx_vec
```

| 方式 | 集める対象 | 選び方 | run コマンド | exp タグ末尾 | 評価 scp |
|---|---|---|---|---|---|
| same | 同一話者の過去 | 直前N | `context_scope=same N=2 ./run_ctxvec.sh` | `-ctx_same_n2` | `ctx_vec_same_n2.scp` |
| session | 両話者の過去 | 直前N（時刻） | `./run_exp_pastctx_session.sh` | `-ctx_session_n2` | `ctx_vec_session_n2.scp` |
| cluster | 同一セッションの過去 | 類似上位 top_k | `./run_exp_pastctx_cluster.sh` | `-ctx_cluster_top5` | `ctx_vec_cluster_top5.scp` |

- 方式ごとに `ctx_vec_<variant>.{ark,scp}` を**別名で出力**（衝突しない）。学習直前に `ctx_vec.scp` へ複製して有効化。
- 評価は方式別 scp を `--ctx_scp` で渡す（§4）。

---

## 3. （参考）初期に試した別案：音声を直接連結（不採用）
`run_pastcontext.sh` は **過去発話の音声を現発話の前に連結**し、ASR 正解も連結する方式だった。
- 問題：ASR の目的が「窓全体を転写」に変わり **CER が比較不能**、系列が長く重い、再転写は無駄。
- → **特徴ベクトル方式（§2）に変更**。`run_pastcontext.sh` は使わない（履歴として残置）。

---

## 4. 実行方法

```bash
cd ~/2025/espnet/egs2/cejc/asr1
context_scope=same N=2 ./run_ctxvec.sh     # (A) 同一話者・直前2発話
./run_exp_pastctx_session.sh               # (B) 両話者・直前2発話
./run_exp_pastctx_cluster.sh               # (C) 過去の類似発話 top_k=5
```
- GPU1枚なので順番に。各々：(1) `ctx_vec_<variant>` 抽出 → `ctx_vec.scp` へ複製、
  (2) `./asr.sh --stage 10`（collect-stats → 学習 → **デコード → スコアリング**）。
- stats は方式別 `exp/asr_stats_raw_jp_word_ctx_<variant>` に分離。共有ストレージ不要（tight dump 利用）。

### 評価（発話区間末タグ：prec/recall/F1）
方式別の scp を `--ctx_scp` で渡す（§2.5 表）：
```
E=exp/asr_20260626-turntaking-noVA-classweight-ctx_session_n2   # 例: session
python local/turntaking/evaluate_multitask.py --config $E/config.yaml \
    --model $E/valid.tag_acc.best.pth --data_dir dump/raw/eval \
    --ctx_scp dump/raw/eval/ctx_vec_session_n2.scp
# cluster は ctx_vec_cluster_top5.scp / same は ctx_vec_same_n2.scp
```
- **ASR 精度（CER/WER）**: 下記 §5（run.sh 実行で自動生成）。

---

## 5. ASR 精度の測り方（stage 12 以降・全実験共通）

asr.sh の標準の流れ（**ref/hyp を置いて sclite でスコア**）をそのまま使う。各 run.sh は
`--stop_stage` を切っていないので、**学習（stage 11）の後に自動で stage 12 デコード → stage 13 スコアリング**まで走る。

```
stage 12  デコード   : test_set を transducer ビームサーチで認識（推論bin = turntaking_asr_inference）
stage 13  スコアリング: ref.trn / hyp.trn を作り sclite で CER/WER を計算
出力     : exp/asr_<asr_tag>/<decode_dir>/<test_set>/score_cer/result.txt （wer も同様）
```
- **turntaking モデルでも ASR デコードは標準どおり**（区間末ヘッド/ctx は ASR デコードでは未使用）。
  そのため `espnet2/bin/turntaking_asr_inference.py`（asr_inference の薄いラッパー）を用意済み。
- **対象実験（すべて run.sh 実行で result.txt が出る）**:
  | 実験 | run スクリプト | test_set | CER の母数 |
  |---|---|---|---|
  | 未来VAなし/あり（trailing） | `run_turntaking_multitask.sh` | `eval_trail` | trailing 音声を認識 |
  | 過去文脈（特徴ベクトル・tight） | `run_ctxvec.sh` | `eval` | 現発話のみ（**クリーン**） |

### 比較時の注意
- **過去文脈版は ASR が完全に不変**＝CER は素の現発話認識（クリーンに比較可）。
- **未来VA版（trailing）は `eval_trail`（trailing 音声）で認識**するので、`<no>` の trailing 区間の発話が
  挿入誤りになりうる（CER がやや高めに出うる）。全実験を**同一条件で**比べたい場合は、
  全モデルを**tight `eval` でデコード**するとよい（trailing 学習モデルには軽い不整合あり）：
  ```
  E=exp/asr_<tag>
  ./asr.sh --stage 12 --asr_tag <tag> --test_sets eval \
      --asr_config <その実験のconfig> --inference_asr_model valid.loss.ave.pth \
      --use_turntaking_detection true --inference_config myconf/decode_cbs_transducer.yaml
  ```

---

## 6. まとめ
- 過去情報は「**直前N発話の encoder 平均ベクトル**」を**区間末ヘッドに concat**して注入（ASR 不変・未来なし）。
- ctx_vec の作り方は差し替え可能（直前N／類似発話クラスタリング 等）。
- ASR 精度は asr.sh の標準 stage 12→13 で **result.txt（CER/WER）** が自動生成。turntaking 用推論bin も用意済み。
