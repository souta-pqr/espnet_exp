# 早期確定：TEASER 詳細設計と実装

作成日: 2026-07-08
関連: [turntaking_early_decision_method.md](turntaking_early_decision_method.md) / [turntaking_methods_detail.md](turntaking_methods_detail.md)
コード: `local/turntaking/teaser.py`（停止方策＋評価）／ `espnet2/asr/espnet_model_turntaking.py`（任意: プレフィックス学習）

> TEASER（Two-tier Early and Accurate Series classifiER, Schäfer & Leser 2020）を発話区間末に適用。
> **「各長さでの予測」＋「その予測が信頼できるかの判定」を二段構えにし、信頼が v 回連続したら確定**する。

---

## 0. 何をするか
現発話を頭から少しずつ聞き、**プレフィックス長 s ごとに区間末タグ確率 p_s** を出す。
各 s で「この予測は信頼してよいか」を **one-class 判定器**が accept/reject し、
**accept が v 回連続**した時点で確定（halt）。早すぎる fluke 確定を防ぎつつ早く確定する。

```
プレフィックス:  s1   s2   s3   s4   s5 ... （聞いた割合 20%,30%,...,100%）
予測 ŷ_s      :  完   完   継   完   完
確率 p_s      :  …    …    …    …    …
信頼判定      :  ✗    ✓    ✗    ✓    ✓(v=2で確定)
                                └─v=2連続─┘→ ここで「完了」確定, earliness = s5 の割合
```

---

## 1. 二段構えの中身
### ① 予測器（各長さのクラス確率 p_s）
- 学習済みマルチタスクモデルの区間末ヘッドを、**encoder 出力を長さ s まででプールして**評価 → p_s。
- 実装：フル音声を1回 encode → encoder出力(T×256) を得て、各 s = int(r·T)（r=0.2,0.3,…,1.0）で
  `p_s = softmax( head( encoder出力[:, :s], 長さ=s, ctx_vec ) )`。1回のencodeで全プレフィックスを安く出す。

### ② 信頼判定器（one-class・各長さごと）
- 各プレフィックス長 s ごとに **one-class 分類器（OneClassSVM）** を用意。
- **学習データで "正解した" 時の p_s ベクトル**（argmax p_s == 正解）だけで fit。
  → 「正解時の確率分布」の内側なら accept、外れ値なら reject。
- テスト時：p_s を判定器 s に通し accept/reject。

### 停止規則（v 回連続 accept で確定）
- s を小さい方から見て、**accept が v 回連続**したら、その時の予測 ŷ_s を確定。earliness = s/T。
- どの s でも v 連続しなければ、**最後の s（全体）で確定**（＝従来の「最後まで聞く」に一致）。
- **v** は dev で「**早さ×精度の調和平均 HM**」を最大化するよう選ぶ（大きいほど慎重＝遅く高精度）。

---

## 2. 予測器を「プレフィックスに強くする」学習（任意・推奨）
既存モデルはフル発話でプールして学習しているため、短いプレフィックスでの p_s は精度が落ちがち。
**学習時に毎回ランダムなプレフィックス長でプール**して最終タグを教師付け（deep supervision）すると、
どの長さでも当たる予測器になり、TEASER の早さが伸びる。
- `espnet_model_turntaking.py`：`tag_prefix_train=True`, `tag_prefix_min=0.3`
  → 学習時、区間末ヘッドのプール長を各サンプル `[0.3,1.0]×T` からランダムに選ぶ（ASR はフルで不変）。
- これを付けたモデルで TEASER を回すと良い。**付けなくても既存モデルにそのまま TEASER 適用は可能**
  （まずは既存の手法1モデルで動作確認 → 必要なら prefix 学習版を用意）。

---

## 3. 全体の流れ（推論）
```
 現発話音声 → 共有エンコーダ（1回）→ encoder出力(T×256)
        │
        ├─ 各プレフィックス s: head で pool → p_s（+ctx_vec）
        ▼
   {p_1, p_2, …, p_S}
        │
        ▼
   ① argmax → ŷ_s   ② one-class 判定器_s → accept/reject
        │
        ▼
   v 回連続 accept の最初の s で確定 → (ŷ, earliness=s/T)
```

## 4. 評価指標
- **精度**：確定タグ ŷ の acc / macro-F1 / クラス別（従来と同じ土俵）。
- **早さ earliness**：確定 s / 発話長 T（小さいほど早い）。
- **総合 HM**：`HM = 2·acc·(1−earliness) / (acc + (1−earliness))`（TEASER 原著の調和平均）。
- v を変えて **精度 vs 早さ曲線**を描く。上限は「最後まで聞いた版（手法1）」。

## 5. 実装（ファイルと役割）
- `local/turntaking/teaser.py`
  1. `--fit_dir`（例 dump/raw/train_dev）で各 s の {p_s} を計算 → **正解サンプルで one-class SVM を s ごとに fit**。
  2. `--eval_dir`（dump/raw/eval）で {p_s} を計算 → **v 連続 accept で halt** → 確定タグと earliness。
  3. v を候補から dev で選ぶ（`--v` 固定も可）。精度/F1/earliness/HM を出力。
  - 既存の turntaking モデル（config+model）にそのまま適用可。ctx_vec 使用時は `--fit_ctx_scp/--eval_ctx_scp`。
- `espnet2/asr/espnet_model_turntaking.py`（任意）：`tag_prefix_train` でプレフィックス学習。

## 6. 実行例
```bash
# 既存の手法1（same N=2）モデルに TEASER を適用（prefix学習なしでも可）
E=exp/asr_20260626-turntaking-noVA-classweight-ctx_same_n2_p30
python local/turntaking/teaser.py \
    --config $E/config.yaml --model $E/valid.tag_acc.best.pth \
    --fit_dir dump/raw/train_dev --fit_seg data/train_dev \
    --eval_dir dump/raw/eval    --eval_seg data/eval \
    --prefixes 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --v auto
```

## 7. 注意（過去の知見）
- 早期は情報が少なく上限が低い（以前 0.62 頭打ち）。**過去文脈 ctx_vec の注入**で早期の p_s を底上げ、
  **v で早すぎ確定を抑制**、が要。TEASER の効果は「**精度をあまり落とさず earliness をどこまで下げられるか**」で測る。
