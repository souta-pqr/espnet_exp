# turn-taking マルチタスク（本命b）実装と評価結果

作成日: 2026-06-15
関連: [turntaking_status_overview.md](turntaking_status_overview.md)（地図）/ [turntaking_vap_approach.md](turntaking_vap_approach.md)（設計）/ [methodA_teaser_why_failed.md](methodA_teaser_why_failed.md)（0.62頭打ちの診断）/ [turntaking_handoff.md](turntaking_handoff.md)（実行ガイド）

> 本資料は「freeze せず conformer encoder を共有してマルチタスク 1 回学習する本命版（status_overview §6-b）」を
> 実装し、eval で評価した結果の記録。**結論: methodA の no/yes 0.62 の天井を突破（0.717）した。**

---

## 1. 何を実装したか

凍結 ASR + 後付け head（前哨実験・`local/turntaking/`）に対し、**ASR を凍結せず encoder を共有**して
ASR（transducer）と区間末予測を同時学習する形にモデル本体を変更した。

```
音声 → contextual_block_conformer encoder（共有・streaming）→ encoder_out (B,T,256)
          │
   ┌──────┴───────────────────────────┐
   ▼                                   ▼
 transducer decoder（既存）        TurnTakingHead（frame-level・因果conv 3層, 0.28M）
   → loss_transducer                ├─ cls (B,T,3): no/yes/other
                                    └─ va  (B,T,4): 未来VA（自己教師）
 総損失 L = loss_transducer + α·loss_tag + β·loss_va     （α=0.3, β=0.2）
```

### 主なファイル
| ファイル | 役割 |
|---|---|
| `espnet2/asr/espnet_model_turntaking.py` | `TurnTakingESPnetASRModel`。encoder 共有・frame-level head・時間重みCE・未来VA |
| `espnet2/tasks/asr.py` | `turntaking` モデル登録 + `TurnTakingASRTask`（tag_label / va_boundary を受ける） |
| `espnet2/bin/turntaking_asr_train.py` | 学習 bin |
| `myconf/train_asr_turntaking_conformer.yaml` | 学習設定（model: turntaking） |
| `local/make_trailing_segments.py` / `local/build_trailing_datadir.sh` | trailing 前処理（segments を区間末+cap に延長、va_boundary 生成） |
| `asr_trailing.sh` | 前処理ラッパー（`<set>_trail` 構築 → ./asr.sh 委譲） |
| `run_turntaking_multitask.sh` | 起動スクリプト |
| `local/turntaking/evaluate_multitask.py` | 学習済みモデルの区間末評価（最終フレーム指標 + TEASER/SPRT） |

### 設計の肝（methodA の診断を反映）
- **frame-level**（CIF をやめる）: CIF は沈黙で発火せず区間末の決定打「無音/次ターン」を観測できないため。
- **trailing 確保**: 各区間を録音から `[start, 元区間末+cap(1.2s)]` で読み直す（`asr_trailing.sh` が segments を延長）。
  ASR のテキストは `[start, 元区間末]` のままなので ASR には無害。
- **時間重み w(t)=floor+(1-floor)·(t/T)**: 序盤の誤りは軽く、終盤ほど重く罰する（早期性と精度のトレードオフを学習に内包）。
- **未来VA（自己教師）**: `va_boundary`（区間末の元秒数）から `target[t,k]=1 if t·frame_sec+h_k < boundary else 0` を model 内合成。

---

## 2. 学習の経過と既知のつまずき

- データ: `train_nodup_trail`（19.7万）/ `train_dev_trail` / `eval_trail`（cap=1.2s）。ASR はゼロから学習。
- 43 epoch で valid/loss が伸びず早期終了（max_epoch=50, patience=10）。約25分/epoch、GPU 22.9/24GB。
- **クラッシュ①（epoch 末の OOM）**: valid 時の transducer ビームサーチ（CER/WER 計算）でメモリ爆発。
  → `report_cer/report_wer: False` で valid のビームサーチを停止。CER は最終デコード stage で測る。
- **クラッシュ②（学習後の平均化）**: `best_model_criterion` を途中変更した副作用で、nbest 平均が削除済み epoch を要求し失敗。
  → 現存 ckpt だけで `valid.tag_acc.ave_10best.pth` を再生成。**再発防止: 最初から 2 基準を設定しておけば pruning が両基準の必要 epoch を保持する。**
- best 基準は CER 非依存に変更: `valid.loss.best`（ASR寄り）と `valid.tag_acc.best`（区間末ベスト）の 2 本。

---

## 3. 評価結果（eval_trail 23,444 区間 / `valid.tag_acc.best.pth`=epoch41）

### 3-1. 最終フレーム予測（早期確定なし）
| 指標 | 値 |
|---|---|
| 3クラス acc | 0.667 |
| macro-F1 | 0.641 |
| **no/yes 二値分離** | **0.717**（対象 17,161 区間） |
| recall: no / yes / other | 0.419 / 0.827 / 0.635 |
| prec: no / yes / other | 0.584 / 0.644 / 0.807 |

混同行列 [真→予測] no/yes/other:
```
<no> : [2621, 3295,  339]
<yes>: [1272, 9017,  617]
<oth>: [ 593, 1698, 3992]
```

### 3-2. methodA との比較（本命の評価軸）
| 設定 | no/yes 分離 | rec_no |
|---|---|---|
| CIF-TEASER（タイト切り・凍結） | 0.62（頭打ち） | 0.237 |
| trailing 線形プローブ（凍結・診断 §9） | 0.70（上限見積り） | — |
| **本手法（trailing + encoder共有マルチタスク）** | **0.717** | **0.419** |

→ **0.62 の天井を突破し、診断の 0.70 見積りもわずかに上回った。** rec_no も 0.237→0.419 とほぼ倍増。
仮説「区間末以降の観測（trailing）＋ encoder を区間末にも最適化（マルチタスク）で天井が上がる」を実証。

### 3-3. 早期確定の動作曲線
TEASER（v=2）:
| θ | acc | rec_no | rec_yes | rec_oth | 確定位置% |
|---|---|---|---|---|---|
| 0.50 | 0.622 | 0.074 | 0.920 | 0.650 | 4.7% |
| 0.60 | 0.653 | 0.141 | 0.895 | 0.741 | 11.5% |
| 0.70 | 0.670 | 0.245 | 0.870 | 0.747 | 26.5% |
| 0.80 | 0.683 | 0.361 | 0.850 | 0.714 | 54.4% |
| 0.90 | 0.681 | 0.419 | 0.832 | 0.680 | 81.1% |

SPRT（v=2）:
| bound | acc | rec_no | rec_yes | rec_oth | 確定位置% |
|---|---|---|---|---|---|
| 1.0 | 0.611 | 0.098 | 0.911 | 0.601 | 3.4% |
| 4.0 | 0.620 | 0.154 | 0.894 | 0.607 | 11.5% |
| 8.0 | 0.642 | 0.190 | 0.891 | 0.659 | 20.8% |
| 16.0 | 0.669 | 0.235 | 0.891 | 0.718 | 36.0% |

→ θ/bound を上げる（＝待つ）ほど **rec_no が単調に改善**（0.07→0.42）。
「<no>（まだ続く）は無音/次ターンを待たないと確定できない」という構造がそのまま曲線に出ており、
「易しい yes は早期確定・難しい no は待つ」という設計意図が機能している。

---

## 4. 残課題と次の打ち手

- **弱点**: 3クラス acc 0.667 は中庸。モデルは多数派 `<yes>`（データの46%）に寄り、真 no の約56%を yes と誤判定。
- **打ち手①（本命）**: `tag_class_weight: [1.92, 1.0, 1.63]`（config でコメントアウト中）を有効化して再学習し、
  `<no>` recall の底上げを図る。
- **打ち手②**: 未来VA は valid loss_va が不安定だった（train≈0.1 / valid≈2）。`va_weight` を 0.05 程度に下げる/切る検証。
- **打ち手③**: 平均モデル `valid.tag_acc.ave_10best.pth` でも同評価し単体ベストと比較。

### 評価の再実行コマンド
```bash
E=exp/asr_20260613-turntaking-multitask-conformer
python local/turntaking/evaluate_multitask.py \
    --config $E/config.yaml --model $E/valid.tag_acc.best.pth \
    --data_dir data/eval_trail
```

### 成果物（checkpoint）
| ファイル | 中身 |
|---|---|
| `valid.tag_acc.best.pth`（→41epoch, valid tag_acc 0.698） | 区間末ベスト単体 |
| `valid.tag_acc.ave_10best.pth` | 区間末上位10平均（再生成） |
| `valid.loss.ave_10best.pth` / `.ave.pth` | ASR寄り上位10平均 |

---

## 5. 再学習: クラス重み版（2026-06-16, exp=`asr_20260615-turntaking-multitask-classweight`）

`<no>` の取りこぼし対策として `tag_class_weight: [1.92, 1.0, 1.63]` を有効化し、**ゼロから再学習**
（va_weight=0.2 据え置き、他条件は同一）。50 epoch 完走。今回は best 基準を最初から 2 本
（valid.loss / valid.tag_acc）にしたため、学習後の平均化クラッシュも発生せず。

### 評価比較（eval_trail 23,444 区間 / `valid.tag_acc.best`、dump 音声で評価）
| 指標 | 重みなし(ep41) | クラス重み版(ep31) | 差 |
|---|---|---|---|
| 3クラス acc | 0.667 | 0.686 | +0.019 |
| macro-F1 | 0.641 | **0.683** | +0.042 |
| no/yes 二値分離 | 0.717 | 0.706 | −0.011 |
| recall `<no>` | 0.419 | **0.584** | +0.165 |
| recall `<yes>` | 0.827 | 0.696 | −0.131 |
| recall `<other>` | 0.635 | 0.770 | +0.135 |
| F1 `<no>` | 0.488 | 0.555 | +0.067 |

クラス重み版 混同行列 [真→予測] no/yes/other:
```
<no> : [3652, 2134,  469]   ← 正解(no)が最多に（前回は yes 流出が最多）
<yes>: [2530, 7595,  781]
<oth>: [ 714,  730, 4839]
```

### 解釈
- **狙いどおり `<no>` 取りこぼしを改善**（recall 0.42→0.58、F1 0.49→0.56）。macro-F1 も 0.64→0.68 と向上＝バランス改善。
- 代償として `<yes>` recall は低下（重みで no 寄りに動かしたため）。
- **no/yes 二値分離はほぼ横ばい（0.717→0.706）**。クラス重みは判定境界を動かして誤りを再配分するもので、
  分離の天井（情報量）自体は上げない。両モデルとも 0.62 を超え trailing 診断の ~0.70 水準を維持。
- 早期確定（TEASER）でも、低 θ での rec_no が 0.07→0.40 に改善し、**早期でも no を拾えるバランス**になった。
  「システムがターンを取るべきでない(no)」を早く検出したい用途では重み版が実用的。

### 注意（運用メモ）
- 原本コーパス共有 `/autofs/diamond2/share/corpus/CEJC_safia` は 2026-06-16 時点でマウントが外れている。
  評価は `dump/raw/<set>` のローカル音声（segment 済み・trailing 込み）で実行する：
  ```bash
  E=exp/asr_20260615-turntaking-multitask-classweight
  python local/turntaking/evaluate_multitask.py --config $E/config.yaml \
      --model $E/valid.tag_acc.best.pth --data_dir dump/raw/eval_trail
  ```
  `evaluate_multitask.py` は segments が無ければ wav.scp を utt 単位として読む（共有不要）。
  再学習や再 dump が必要なときは共有の再マウントが要る。
