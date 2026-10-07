# 発話区間末予測：現状整理と今後の方針（混乱を解くための地図）

作成日: 2026-06-13（同日 §5 を訂正、§3 にドメイン知識を追記）
目的: 「もともと何をやっていたか」「何で行き詰まったか」「今どの選択肢にいるか」を一枚にまとめる。
関連: [cif/00_index_comparison.md](cif/00_index_comparison.md) / [methodA_teaser_why_failed.md](methodA_teaser_why_failed.md) / [turntaking_vap_approach.md](turntaking_vap_approach.md)

---

## 0. 解きたい問題（これは一貫して変わっていない）

> ユーザの発話の途中で「システムがターンを取るべきか」を、**精度を落とさずできるだけ早く**確定したい。
> ラベルは区間単位で 3 種：`<no>`=区間末でない / `<yes>`=区間末 / `<other>`=相槌。
> 易しい発話は早く決め、難しい発話はしっかり待つ。評価は accuracy と earliness（早さ）の両立。

---

## 1. 全体の流れ（時系列）

```
 ①もともと: CIF-TEASER          ②診断で行き詰まり        ③打開策の検討            ④今ここ
 ─────────────────         ───────────────       ─────────────────       ────────────────
 CIF発火ごとにタグ分類    →   no/yes が0.62で頭打ち  →  (a)凍結ASR+後付けhead  →  freezeしない
 + 停止規則(A〜E)で早期確定    = 情報不足と判明          (b)マルチタスク化         マルチタスク1回学習
 ※encoderは conformer が基本                                                     ＝conformerで実装
```

---

## 2. もともとのモデル ＝ CIF-TEASER（と手法 A〜E）

### 共通アーキテクチャ（cif/00_index_comparison.md より）
```
音声 → Encoder(Contextual Block Conformer) → CIF機構 ─┬→ Decoder → テキスト(ASR)
        ※もともとの実装は conformer            └→ tag_classifier(MLP D→128→3) → p_i(no/yes/other)
                                                       └ 早期確定ロジック（手法ごと）
```
- **ベースは「CIF 付き ASR」**。CIF＝音響を**トークン単位の発火(fire)**に積算する機構。
- 各 fire で `tag_classifier`（小さな MLP）が no/yes/other の確率 `p_i` を出す。
- **判定単位は CIF の fire**（≈1トークン分）。tag_classifier までは手法 A〜E 全部で共通。

### 「いつ確定するか」の停止規則が手法 A〜E（ここだけが違う）
| 手法 | 種別 | 一言 |
|---|---|---|
| A: TEASER | 確信度・局所 | master(stop_head) が「今信頼してよいか」を学習＋同クラス v 回連続で確定 |
| B: ECONOMY | コスト | 期待コストが将来最小になる点で確定 |
| C: SPRT | 累積・大域 | 対数尤度比を累積し閾値超で確定（**学習不要**・ベースライン） |
| D: RL halting | 学習方策 | 停止方策を強化学習で end-to-end 学習 |
| E: 損失整形 | 横断改良 | tag_classifier 自体を早期化（A〜C を底上げ） |

> 「**もともとの発話区間末予測**」は:
> **CIF 付き ASR の encoder 表現を、CIF 発火ごとに tag_classifier で 3 クラス分類し、
> 停止規則（A〜E）でいつ確定するかを決める**もの。
> TEASER(手法A) は `tag_classifier`(slave) を固定してから `stop_head`(master) を学習する 2 段階方式。

---

## 3. 区間末予測の手がかり（ドメイン知識）

ローカル LLM に**テキストのみ**で「区間末か否か」を判定させたところ、**約8割**当たった。
今回は**音声（韻律）も加わる**ので、それ以上を狙える、というのが見立て。LLM に与えた判定基準:

- **(a) 独り言**：相手に向けず自己完結（例「あれ、どこ置いたかな」）
- **(b) 明確な質問・問いかけ**：疑問語や「〜の?」で相手に反応を求める
- **(c) ターンを譲りうる文末形式**：「〜よね/〜だよね/〜でしょ/〜じゃない/〜じゃん/〜かな/〜みたいな」等
- **(d) 話題の大きな切り替わり**：直前に長いポーズ（目安2秒以上）があり話題が転換
- → (a)〜(d) のどれにも当てはまらない（「〜て/〜で/〜けど/〜から/〜し」等の接続・中止形、
  フィラー(F)・言い直し(D)で作りかけ）なら「区間末でない」。

> **重要な含意**：(a)(b)(c) は**文末の言語形式**＝テキスト情報、(d)とポーズは**韻律・タイミング**情報。
> つまり区間末は「**文末の言語形式が主**（テキストで8割）＋**韻律・ポーズで上積み**（音声）」という構造。
> → ASR が持つ**言語的表現**が効くタスク。テキスト＋音響を併せ持つ ASR encoder は、
>   テキストのみの LLM より有利になり得る（音声で上積みできるという見立ての根拠）。
> → 逆に「純粋な音響特徴だけ」では言語形式の手がかりを取りこぼす懸念があり、
>   **encoder に区間末も学ばせるマルチタスク**が効く理由になる（§6・§7）。

---

## 4. 何で行き詰まったか（診断結果・methodA_teaser_why_failed.md）

`<other>`(相槌) は F1 0.93 で解けたが、**`<no>` と `<yes>` の区別が ~0.62 で頭打ち**になった。
クラス重み・文脈集約・encoder プローブを試したが全部 0.60〜0.62。

**根本原因＝表現力不足ではなく「情報不足」**:
1. **CIF は沈黙で発火しない** — 区間末の決定打「発話直後の無音／次話者ターン」を観測できない（最終語で発火が尽きる）。
2. **セグメントが語末で切られている** — 末尾無音は中央値 0ms。区間末以降の無音が物理的に存在しない。
3. **trailing 実験**: 各区間を録音から `[start, end+Δ]` で読み直して区間末以降まで観測すると、
   no/yes 分離は Δ=0→0.62、Δ=800ms→0.68、Δ=1200ms→0.70 と**単調改善**。
   → 0.62 は「streaming の限界」ではなく「セグメント化＋即決のアーティファクト」。**待てば上がる**。

（補足：この trailing 実験は transformer 版 exp で測ったが、見ているのは**区間末予測そのものの性質**
 なので conformer でも同様に当てはまる。§5 参照。§3 のドメイン知識＝言語形式＋ポーズとも整合する。）

---

## 5. 【訂正】区間末予測は encoder に依存しない（もともと conformer）

> 旧版で「区間末予測=transformer 系統 / conformer=別の運用 ASR」と 2 系統に分けたのは**誤り**でした。
> 区間末予測（CIF + tag_classifier の枠組み）は **encoder アーキテクチャに依存しない**タスクで、
> **もともと conformer ベースで実装**していた。transformer はあくまで診断で使った一例にすぎない。

| encoder | 位置づけ |
|---|---|
| **Contextual Block Conformer** | **もともとの実装。今後もこれで実装する**（streaming, block18/hop3/look3） |
| Contextual Block Transformer | `asr_cif-teaser-methodA-v2-stage1`。§4 の診断で使った一例にすぎない |

- タグ枠組み自体は両 encoder に載る。encoder 違いは「研究 vs 運用」の対立**ではない**。
- §4 の trailing 診断（0.62→0.70）は transformer で測ったが、conformer でも同様に当てはまる。
- **今後は conformer（contextual_block_conformer）で実装・再計測する。**

---

## 6. 打開策の比較（③で検討した 2 案）

| | (a) 凍結ASR + 後付けhead | (b) マルチタスク（freezeしない）★方針 |
|---|---|---|
| 学習 | ASR は凍結。小さな head だけ別学習 | **1 回**で ASR と区間末タグを同時学習 |
| encoder | 区間末用に最適化**されない** | 区間末用にも**最適化される**（§3 の言語形式を学べる） |
| no/yes 精度天井 | ~0.70 で頭打ち | **天井を破れる可能性** |
| ASR への影響 | ゼロ（不変） | 変わりうる（損失バランス要調整） |
| この環境での実行 | **可能**（head だけ軽量） | **不可**（ASR 再学習が RAM 31GB で OOM。別環境が要る） |

**判定単位は両案とも CIF をやめて frame-level**（毎フレーム連続出力）。CIF は沈黙で発火しない（§4-1）ため。

---

## 7. 今後の方針（conformer・frame-level）

```
音声 ─► Contextual Block Conformer encoder（共有・streaming）─► encoder_out
                              │
          ┌───────────────────┼────────────────────────┐
          ▼                                             ▼
   ASR: decoder（既存のまま）              turn-taking head（frame-level・因果conv）
        → loss_asr                            ├─ cls (T,3): 継続/区間末/相槌
                                              └─ va  (T,B): 未来VA（自己教師）
   総損失  L = loss_asr + α·loss_tag + β·loss_va     （マルチタスク本命）
   推論: frame-level の cls 確率に停止規則(TEASER/SPRT)を適用して早期確定
```

### 段取り（現実的な順序）
1. **まずこの環境で**：既存 conformer（`exp/asr_0417-...`）を **freeze** + frame-level head を学習（前哨実験）。
   - §4 の trailing は head 側が録音から `[start, end+1.2s]` を読んで確保（re-dump 不要）。
   - これで「frame-level＋trailing＋conformer」で 0.62 を超えるかを安く検証。
2. **別環境で**：ASR をマルチタスク（conformer encoder 共有で ASR＋区間末を1回学習）。本命。

### 実装上の重要メモ（つまずきポイント）
- **0417 のロードは `espnet2.tasks.asr.ASRTask`** を使う（`run_streaming.sh` は `asr_task=asr` で学習）。
  `ASRTransducerTask`（espnet2/asr_transducer/）では encoder が噛み合わず失敗する。← 実際にここで詰まった。
  `local/turntaking/extract_features.py` のモデルロードを `ASRTask.build_model_from_file(...)` にする。
- encoder output_size=256 ＝ head の `d_in=256` と一致。
- ストリーミング制約（look_ahead=3）は encoder 共通なので両タスクで自動的に両立。
- 損失重み α,β は ASR 優先（小さめ）から。

### まだ決める論点
- trailing 確保：マルチタスク版では segments を区間末+1.0〜1.2s に延長して dump し直すか（ASR には無害）。
- マルチタスクの実装方式：既存 conformer model を派生させて turn-taking ブランチ＋損失を足す形。

---

## 8. 関連資料の地図
- [cif/00_index_comparison.md](cif/00_index_comparison.md) … もともとの手法 A〜E の全体像
- [cif/method_A_TEASER.md](cif/method_A_TEASER.md) … TEASER(手法A) の詳細設計
- [methodA_teaser_why_failed.md](methodA_teaser_why_failed.md) … なぜ 0.62 で頭打ちか（診断・trailing 実験）
- [methodA_teaser_diagnosis.md](methodA_teaser_diagnosis.md) … 診断の全数値
- [turntaking_vap_approach.md](turntaking_vap_approach.md) … (a)凍結 frame-level 案の設計（今回(b)に発展）
- 本資料 … 現状整理と今後（conformer・frame-level、本命はマルチタスク）
