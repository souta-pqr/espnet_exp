# 別手法：RL halting policy（学習型停止方策）

作成日: 2026-06-16
関連: [turntaking_multitask_method.md](turntaking_multitask_method.md)（本体モデル：frame-level 区間末ヘッド）/ [turntaking_related_problems.md](turntaking_related_problems.md)（系列の早期分類・系統①〜③）
コード: `local/turntaking/rl_halting.py` / 実行: `run_rl_halting.sh`

> 本資料は「**いつ確定するか（停止）を強化学習で獲得する**」手法をまとめたもの。
> TEASER/SPRT（後付けの固定規則＝系統①）とは異なり、停止タイミング自体をモデルが学ぶ（系統②）。
> 解きたい問題は本体と同じ：**意味的発話区間末（no/yes/other）を、できるだけ早期に・誤判定なく確定**する。

---

## 0. なぜこの手法か（TEASER/SPRT との違い）

- 本体モデルは毎フレーム確率 `p_t = [P(no), P(yes), P(other)]` を出す（[method](turntaking_multitask_method.md)）。
- TEASER/SPRT は **その p_t に後付けする固定規則**で、しきい値 θ・bound を人が dev で sweep して決める（系統①）。
- RL halting は **「停止か継続か」を、精度と早さの報酬から直接学習**する（系統②）。
  しきい値設計が要らず、停止タイミングをモデルが獲得する。当初の定式化
  「**遅延ペナルティ＋早すぎ誤判定ペナルティを損失に明示する**」に最も忠実な形（文献の EARLIEST/ACT 系）。

> 比較は公平：**同じ p_t の上**で「精度 × 確定位置%」を測るので、固定規則（①）と学習型停止（②）を直接比べられる。

---

## 1. 仕組み（frame-level）

```
   音声 [start, end+cap]
        │
   共有エンコーダ → encoder_out
        │
  ┌─────┴──────────────────────────┐
  │ Discriminator = 区間末ヘッド(cls) │  ← 学習済みを流用（凍結）
  └─────┬──────────────────────────┘
        │  p_t = softmax(cls_logits_t)   (毎フレームの no/yes/other 確率)
        ▼
  ┌──────────────────────────────┐
  │ Controller（小MLP・強化学習）   │  h_t を入力に halt_prob_t を出す
  └─────┬────────────────────────┘
        │  halt_prob_t
        ▼
   ◇ 各フレームで「停止？」
     ├ Yes（halt）→ その時刻 τ のクラス argmax(p_τ) で確定
     └ No        → 次のフレームへ
```

- **Discriminator**：本体の発話区間末ヘッド（cls）。`p_t` を出す。ここは**学習済みを凍結して流用**する
  （「停止を学習する効果」だけを切り分けるため。フル end-to-end にしたい場合は §5）。
- **Controller**：各フレームで**停止確率** `halt_prob_t` を出す小さな MLP。これを強化学習で獲得する。

---

## 2. 数式

**Discriminator（区間末ヘッド・凍結）**

p_t = softmax( cls_head(h_t^enc) )      （h_t^enc は encoder_out から区間末ヘッドが作る特徴）

**Controller（停止方策）**

halt_prob_t = σ( MLP( s_t ) )

s_t = [ p_t,  (top1 − top2),  entropy(p_t),  t/(T−1) ]   （= 6 次元）

```
- halt_prob_t : フレーム t で停止する確率（0〜1）
- s_t         : Controller への状態入力（その時刻までで計算でき、未来を使わない）
    ・p_t            : 区間末ヘッドの 3 クラス確率
    ・top1 − top2    : 確からしさの差（確信度マージン）
    ・entropy(p_t)   : 迷いの大きさ
    ・t/(T−1)        : 発話内の相対位置（0=最初, 1=最後）
```

**報酬**

R = r_correct( ŷ_τ , y ) − λ · ( τ / (T−1) )

r_correct( ŷ_τ , y ) = +1 （ŷ_τ = y：正解） / −c （ŷ_τ ≠ y：不正解）

```
- τ : 停止したフレーム（ŷ_τ = argmax p_τ で確定）
- λ : 遅延ペナルティの重み（大きいほど早く止まることを優先）
- c : 不正解ペナルティ（c>1 で誤りをより強く罰する）
```

> 報酬は「**正しく当てれば +1、間違えれば −c、遅れた分だけ −λ·遅延**」。
> これを最大化するよう Controller を学習する＝「精度を保てる範囲でできるだけ早く止める」を直接最適化。

---

## 3. 学習方法（REINFORCE）

停止は微分不可能な離散行動なので、**方策勾配（REINFORCE）**で学習する。

```
各発話（=1エピソード）について:
  1) 各フレームで halt_prob_t を計算し、停止/継続をサンプル → 最初に停止した τ を得る
  2) ŷ_τ = argmax p_τ、報酬 R を計算
  3) 軌跡の対数確率 log P = Σ_{t<τ} log(1−halt_prob_t) + log(halt_prob_τ)
  4) 損失 = −(R − baseline) · log P     （baseline=報酬の移動平均で分散低減）
```

- **Discriminator（cls）は凍結**：学習済み p_t を使い、Controller だけを更新する。
- **Controller** のみ REINFORCE で報酬 R を最大化。
- λ を振ると「精度 × 確定位置%（遅延）」の動作点が変わる（TEASER の θ・SPRT の bound に相当する操作）。

---

## 4. 実装と使い方

### コード: `local/turntaking/rl_halting.py`
1. 学習済みモデルを 1 回通して**毎フレーム確率 `p_t` を抽出**（utt 単位・`dump/raw/<set>_trail`・共有マウント不要）。
   → `local/turntaking/rl_cache/` にキャッシュ（以降の λ 振りは高速）。
2. その上で **Controller を REINFORCE 学習**（train: `train_dev_trail`）。
3. **評価**（`eval_trail`）：halt_prob>0.5 で貪欲に停止し、`acc / クラス別 recall / 確定位置%` を出力。

### 実行: `run_rl_halting.sh`
```bash
cd ~/2025/espnet/egs2/cejc/asr1
# 既存の学習済みモデルでそのまま実験できる（フルASR再学習は不要）
asr_tag=20260615-turntaking-multitask-classweight ./run_rl_halting.sh
# λ を変えて動作曲線を見る（既定で 0.1 0.3 0.5 1.0 を sweep）
lambdas="0.2 0.5 1.0 2.0" epochs=30 ./run_rl_halting.sh
```
主な環境変数：`asr_tag`（使うモデル）/ `ckpt`（既定 `valid.tag_acc.best.pth`）/ `lambdas` / `epochs` /
`train_dir` / `eval_dir`。スクリプトはそのまま `rl_halting.py` に引数を渡す。

---

## 5. 位置づけと限界

- 本実装は **Discriminator（cls）を凍結**し、Controller だけを学習する**軽量版**。
  「停止を学習する効果」を、学習済み p_t の上で TEASER/SPRT と公平に比較できる。
- **フル end-to-end**（Discriminator も同時に CE で学習し、Controller を RL で学習）にするには、
  本体モデルに Controller を内蔵して同時最適化する改修が必要（より強力だが学習は不安定になりやすい）。
- REINFORCE は分散が大きいので、baseline・λ・c・学習率の調整が要る。安定しない場合は PPO 等も選択肢。
- 評価軸は本体と同じ「精度 × 確定位置%」。TEASER/SPRT 曲線（`evaluate_multitask.py`）と重ねて比較する。

---

## 6. 一言まとめ

> **停止タイミングを人が設計する（θ/bound）のではなく、報酬「正解 +1／誤り −c／遅延 −λ」で学習する**のが RL halting。
> 学習済みの区間末ヘッド（p_t）はそのまま使い、その上に停止方策（Controller）を REINFORCE で載せる。
> 当初の定式化（遅延罰＋早すぎ誤判定罰を損失に明示）に最も忠実な手法で、TEASER/SPRT と同じ土俵で比較できる。
