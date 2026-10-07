# 過去音声を連結して発話区間末を予測する（seqcat 方式・手法解説）

作成日: 2026-07-06
関連: [turntaking_multitask_method.md](turntaking_multitask_method.md)（本体モデル）/ [turntaking_pastcontext_method.md](turntaking_pastcontext_method.md)（特徴ベクトル方式）
コード: `espnet2/asr/espnet_model_turntaking.py`（context_mode=seqcat）/ `local/build_past_audio.py` / `run_seqcat.sh` / `myconf/train_asr_turntaking_conformer_seqcat.yaml`

> これまで（ctx方式）は「現発話プール」と「過去プール」を**別々に作って concat**していた。
> seqcat 方式では、**過去N発話の音声を現発話の前に一列に連結**し、**共有エンコーダに1回通して**、
> **現発話フレームだけを pooling** して1ベクトルにする。過去はエンコーダの self-attention 経由で
> 現発話フレームに畳み込まれる。**ASR は現発話のみで不変**。

---

## 1. 目的
- 「[N-2, N-1, 現発話] を並べて **1本のベクトル（現発話のベクトル）** を作り、発話区間末タグを出す」構造にする。
- 過去は「別プールを concat」ではなく、**エンコーダの中で現発話に文脈として効かせる**。
- **average pooling は使わない**（transformer 出力の平均は勾配が均一化しやすく相性が悪い）。**max**（既定）／将来 attention。

---

## 2. 構造

```
   [N-2] [N-1] [現発話]  ← 過去N発話の音声を時間連結（現発話が必ず最後）
        │
   共有エンコーダ (1回 forward, Contextual Block Conformer)
        │
   出力フレーム列:  [====過去====][==現発話==]
                                    │
                     ★現発話フレームのみ max pooling★（average は使わない）
                                    │
                               1ベクトル(256)
                                    │
                                   MLP
                                    │
                              発話区間末タグ（継続/完了/相槌）

   （ASR用は別 forward: [現発話]のみ → Encoder → Transducer → テキスト・不変）
```

- **過去の扱い**：pooling では直接使わない。連結してエンコーダに通すことで、self-attention により
  **現発話フレームの出力に過去の情報が畳み込まれる**（＝前置き文脈）。過去フレーム自体は pool から外す。
- **現発話フレームの取り出し**：連結系列のエンコーダ出力の**末尾 n_cur フレーム**（n_cur=現発話のみを
  エンコードした長さ）を現発話とみなす（現発話は連結の最後尾にあるため）。
- **計算量**：N が増えるほど連結系列が伸び、エンコーダ処理が重くなる（設計上の代償）。

## 2.5 過去発話の決め方
- **「現発話の end より前」** に終わる発話を過去とする。
- **同一話者**では end 基準＝start 基準（過去は必ず現発話より前に終わる）＝**直前N発話**。現発話が必ず最後尾。
- 複数話者(session)で end 基準にすると現発話に重なる相手の相槌等も含みうる（並べ方は session 実装時に決定）。
- `max_past_sec`（既定30秒）窓内の直前N発話を採用。会話先頭など過去が無い区間は 20ms 無音（＝実質 現発話のみ）。

---

## 3. データの持ち方（ストレージ増なし）
- `local/build_past_audio.py` が各発話の **past_speech.scp** を作る。
  中身は **sox パイプで過去N発話の flac を連結**（古→新）：`utt  sox f1.flac f2.flac -t wav - |`
- 実体の音声ファイルは新規生成しない（既存 flac をパイプで連結）。
- モデルが past_speech の後ろに現発話 (`speech`) を連結してエンコード（`_concat_wave`→`_seqcat_head`）。

## 4. 実装（変更点）
- `espnet2/asr/espnet_model_turntaking.py`
  - `context_mode: seqcat` / `tt_pool: max` を追加。`_concat_wave`（波形連結）・`_seqcat_head`（連結→encode→現発話フレーム max pool→MLP）。
  - forward に `past_speech` / `past_speech_lengths` を追加。`ctx_dim=0`（ctxベクトルは未使用）。
- `espnet2/tasks/asr.py`：`optional_data_names` に `past_speech` を追加。
- `asr.sh`：`past_speech.scp` があれば `--*_data_path_and_name_and_type …/past_speech.scp,past_speech,sound` を配線。
- `myconf/train_asr_turntaking_conformer_seqcat.yaml`：`context_mode: seqcat`, `tt_pool: max`, `ctx_dim: 0`。

---

## 4.5 pooling 範囲（2通り・両方実験する）
`tt_pool_range` で切替（config `_whole` / `_current`、run では `pool_range` 環境変数）:
- **whole**（既定・まずこちら）：連結系列**全体（過去+現発話）**を max pool。過去も**直接ベクトルに寄与**。
- **current**：**現発話フレームのみ** max pool。過去はエンコーダ self-attention の**文脈としてのみ**効く（§2）。
- past_speech（過去音声）は両者で共有（同じ）。exp/stats/tag のみ `_whole` / `_current` で区別。

## 5. 実行方法（同一話者・N=1〜5）
```bash
cd ~/2025/espnet/egs2/cejc/asr1
tmux new -s sameseqcat
# ① 過去も直接 pool（whole・まずこちら）
pool_range=whole   NS="1 2 3 4 5" max_past=30 ./run_exp_pastctx_same_seqcat_series.sh
# ② 現発話のみ pool（current・そのあと）
pool_range=current NS="1 2 3 4 5" max_past=30 ./run_exp_pastctx_same_seqcat_series.sh
# 単発なら: pool_range=whole scope=same N=2 max_past=30 ./run_seqcat.sh
```
- 各 N：past_speech 構築 → `past_speech.scp` に複製 → collect-stats → 学習 → **デコード（bounded 探索）→ CER**。
- デコードは既定で `myconf/decode_cbs_transducer_bounded.yaml`（メモリ暴走回避の alsd）。
- 出力：exp `asr_<日付>-turntaking-noVA-classweight-ctx_same_n{N}_p30_seqcat`、past `dump/raw/<set>/past_speech_same_n{N}_p30_seqcat.scp`。
- **必ず tmux/nohup 内**（5本で数日・N大ほど連結が長く重い）。GPUは1枚。

## 6. 評価（発話区間末 prec/recall/F1）
```bash
E=exp/asr_<日付>-turntaking-noVA-classweight-ctx_same_n2_p30_seqcat
python local/turntaking/evaluate_multitask.py --config $E/config.yaml \
    --model $E/valid.tag_acc.best.pth --data_dir dump/raw/eval \
    --seqcat --seg_dir data/eval --scope same --n_past 2 --max_past_sec 30
```
- `--seqcat`：過去音声を連結して現発話フレーム pool で予測（学習と同じ選択則）。
- ASR 精度（CER）：run.sh の stage 12→13 で `…/score_cer/result.txt`。

## 7. ctx 方式（特徴ベクトル）との違い
| | ctx 方式（既存） | seqcat 方式（本手法） |
|---|---|---|
| 過去の使い方 | 過去を別に要約(uvec平均)し**concat** | 過去音声を**連結してエンコーダに通す**（文脈として現発話に畳み込み） |
| pooling | 現発話・過去とも average | **現発話フレームのみ max**（average不使用） |
| エンコーダ | 現発話1回 | 現発話1回 ＋ 連結系列1回（**重い**） |
| ctx_dim | 256/80 | 0（連結で表現） |
