# turn-taking（発話区間末予測）実行ガイド & ハンドオフ

最終更新: 2026-06-13
このファイルの役割: **次のセッションで、何があって・どう動かすか・次に何をするか**がすぐ分かるようにする実務メモ。
背景・設計は別資料: [turntaking_status_overview.md](turntaking_status_overview.md)（現状の地図） / [turntaking_vap_approach.md](turntaking_vap_approach.md)（設計）

---

## 0. いま何ができる状態か（一言）

**既存の conformer ASR（0417）を freeze して、frame-level で発話区間末タグ（no/yes/other）を予測する
パイプラインが動く状態。スモーク（各120件）は完走確認済み。本番は未実行。**

本命は「freeze せずマルチタスク（ASR + 区間末を1回学習）」だが、この環境は RAM 31GB で ASR 学習が
OOM になるため、まず凍結版で frame-level + trailing の効果を検証する段階。

---

## 1. 環境（必須）

- conda env: **`/home/kobori/.conda/envs/espnet`** 固定（`/opt/anaconda3` だと `espnet2` import 不可）
- 作業ディレクトリ: `/home/kobori/2025/espnet/egs2/cejc/asr1`
- GPU 1枚（24GB）。turn-taking の抽出/学習は軽い。

---

## 2. ファイル構成と各スクリプトの役割

### turn-taking 本体: `local/turntaking/`
| ファイル | 役割 | 主な引数（既定） |
|---|---|---|
| `model.py` | `TurnTakingHead`: 因果dilated conv3層 + 2ヘッド（cls 3クラス / va 未来VA）。0.28M params、frame-level、未来を見ない | d_in=256, d_hidden=128, n_layers=3, va_bins=4 |
| `extract_features.py` | 凍結 ASR を通して `encoder_out (T,256)` と未来VAラベルを抽出し npz シャード保存。録音から `[start, end+cap]` を読むので**区間末以降(trailing)も観測** | `--task asr`(0417用)/cif/transducer, --config, --model, --wav_scp, --segments, --tag, --out_dir, --cap 1.0, --horizons 0.2,0.4,0.6,1.0, --max_utts 0, --device cuda |
| `train.py` | head 学習。損失 = 時間重み付きクラスCE + λ·未来VA-BCE。dev で last-frame acc を見て best 保存 | --train_dir, --dev_dir, --out, --epochs 15, --batch_size 64, --lr 1e-3, --lambda_va 1.0, --d_hidden 128, --n_layers 3 |
| `evaluate.py` | 停止規則で「精度 vs 早期性」を計測。TEASER（p_max>θ が v 連続）/ SPRT（対数確率累積>bound） | --ckpt, --eval_dir, --rule teaser/sprt, --v 2, --thresholds 0.5..0.9, --sprt_bounds 1..16 |
| `README.md` | 短い説明 |

### オーケストレーション（asr1/ 直下）
| ファイル | 役割 |
|---|---|
| `run_turntaking.sh` | 抽出→head学習→評価 を一括実行。`asr_tag` を渡して既存/学習済み ASR を freeze して使う |
| `run_streaming.sh` | **土台 ASR（contextual_block_conformer, transducer）の学習**（ユーザーの既存スクリプト）。マルチタスク前段や ASR 再学習はこれ |

### 中間生成物
- `local/turntaking/feats/{train_nodup,train_dev,eval}/shard_*.npz` … 抽出キャッシュ（再生成可能。全件で約6GB）
- `local/turntaking/head.pth` … 学習済み head

---

## 3. 実行方法

### 3-1. 既存 0417 conformer を freeze して回す（いま動く道）
```bash
cd /home/kobori/2025/espnet/egs2/cejc/asr1

# スモーク（各120件。動作確認用）
asr_tag=0417-transducer-shoji-cejc-section-func-add MAX_UTTS=120 cap=1.2 ./run_turntaking.sh

# 本番（全件。train_nodup 19.7万件の抽出が重い。一度きりでキャッシュされる）
asr_tag=0417-transducer-shoji-cejc-section-func-add cap=1.2 ./run_turntaking.sh
```

### 3-2. 部分実行（env トグル）
```bash
# 抽出済み → 学習+評価のみ
asr_tag=0417-... run_extract=false ./run_turntaking.sh
# 評価のみ（head 学習済み）
asr_tag=0417-... run_extract=false run_train=false ./run_turntaking.sh
```

### 3-3. 主要パラメータ（env で上書き）
- `cap`（既定1.0, **推奨1.2**）: 区間末より先に読む秒数。trailing（無音/次ターン）を観測するほど no/yes 分離↑。
- `asr_task`（既定 asr）: 0417 等の標準 ASR は `asr`。CIF-TEASER なら `cif`。
- `asr_model` / `asr_config`: 既定は `exp/asr_${asr_tag}/{valid.cer_transducer.best.pth, config.yaml}`。違う場合は直接指定。
- `MAX_UTTS`: 0=全件、>0 で各セットの件数制限（スモーク用）。

---

## 4. 結果の見方

- 抽出: ログに `完了: written=N fail=0` が3セット分出れば成功。`feats/<set>/manifest.txt` ができる。
- 学習: `epoch K  dev_lastframe_acc=...`、best 更新で `saved best -> head.pth`。
- 評価: TEASER/SPRT それぞれ「θ(or bound) / acc / rec_no / rec_yes / rec_oth / 確定位置%」の表。
  - **確定位置% が小さいほど早期確定（遅延小）**。θ/bound を上げると精度↑・遅延↑。
  - 旧 CIF-TEASER baseline は no/yes acc ~0.62、rec_no 0.237。これを超えるかが本実験の見どころ。

---

## 5. つまずきポイント（重要・既知）

1. **0417 のロードは `espnet2.tasks.asr.ASRTask`**（`extract_features.py` の `--task asr`）。
   `ASRTransducerTask`（espnet2/asr_transducer/）を使うと encoder が噛み合わず失敗する。← 一度ここで詰まった。
   0417 は `run_streaming.sh`（`asr_task=asr`）で学習された ASRTask + transducer デコーダのモデル。
2. **ASR 再学習の OOM**: RAM 31GB だと valid 突入時に OOM(code 137)。`valid_batch_bins` を 300000→**50000** に
   下げ済み（`myconf/train_asr_transducer_conformer.yaml` と exp の config.yaml 両方）。それでも厳しいので
   **ASR 学習は別環境推奨**。`report_cer:true` の valid beam search が重い一因。
3. **asr.sh の stage 番号**: このレシピでは **stage 11 = collect stats、stage 12 = ASR Training**。
   学習を resume するなら `./run_streaming.sh --stage 11 --stop_stage 12`（11 経由で 12 へ）。
   `--stage 12` 単独だと training 前段の準備が満たされず素通りすることがある。
4. **frame timing**: 1 frame ≈ 33ms（frontend hop 132/16k ×subsample4）。conformer の look_ahead=3 ≈100ms。
   trailing は `cap` で録音から読むので look_ahead に依存せず確保できる。

---

## 6. 今後すべきこと

### 段取り①（この環境で・いま）: 凍結 0417 で本番検証
```bash
asr_tag=0417-transducer-shoji-cejc-section-func-add cap=1.2 ./run_turntaking.sh
```
- 目的: frame-level + trailing + conformer で no/yes が 0.62 を超えるか確認。
- 超えれば、本命マルチタスクへの投資判断ができる。

### 段取り②（別環境で・本命）: マルチタスク 1回学習
- conformer encoder を ASR と区間末予測で**共有**し、`L = loss_asr + α·loss_tag + β·loss_va` を同時最適化。
- encoder が区間末（§3 のドメイン知識＝文末の言語形式＋韻律）にも最適化され、凍結の天井 ~0.70 を破れる可能性。
- 実装の論点（未決）:
  - turn-taking ブランチを既存 conformer model に内蔵（`model.py` の `TurnTakingHead` を encoder 直後に）。
  - tag ラベルを ESPnet 学習に渡す仕組み（CIF-TEASER に前例: `espnet_model_cif_tag.py`）。
  - **trailing 確保**: segments を区間末+1.0〜1.2s に延長して dump し直す（ASR には無害）。
  - 損失重み α,β は ASR 優先（小さめ）から。
- ASR 再学習を伴うため、RAM に余裕のある別環境で実施。

### ドメイン知識（設計の指針・status_overview §3）
区間末は「文末の言語形式（独り言/疑問/『〜よね』等の譲り表現）＋ポーズ・話題転換」が手がかり。
ローカル LLM がテキストのみで約8割。音声(韻律)を足せば上積み期待 → ASR の言語表現を活かす設計が有利。

---

## 7. 関連資料
- [turntaking_status_overview.md](turntaking_status_overview.md) … 現状の地図（もともと何を→なぜ頭打ち→今後）
- [turntaking_vap_approach.md](turntaking_vap_approach.md) … frame-level 設計の詳細
- [methodA_teaser_why_failed.md](methodA_teaser_why_failed.md) … CIF-TEASER が 0.62 で頭打ちの診断（trailing 実験）
- [cif/00_index_comparison.md](cif/00_index_comparison.md) … もともとの手法 A〜E
- `local/turntaking/README.md` … パイプラインの短い説明

---

## 補足: このセッションでの注意
2026-06-13 のセッションは Read/Grep/Bash の**出力表示が化ける**不具合があり、ファイル編集は Write で全体上書きして対処した。
新しいセッションでは解消している見込み。コード自体（上記スクリプト）は py_compile/bash -n 通過・スモーク完走済みで正常。
