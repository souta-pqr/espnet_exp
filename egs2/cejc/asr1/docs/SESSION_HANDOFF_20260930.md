# 引き継ぎメモ（2026-09-30 時点）

作業ディレクトリ: `~/2025/espnet/egs2/cejc/asr1`
Python: `/home/kobori/.conda/envs/espnet/bin/python`（conda 環境 `espnet`）

**最初に `docs/HANDOFF.md` を全部読むこと。** 特に §2.6（Stage1 改善後の再評価）と
§2.7（テキスト側の打ち止め記録）、および §2.7 直後の「次にやること」。

---

## 1. いまの到達点

発話区間末検出（継続／完了／相槌の 3 クラス）を、音声モデルと
「ASR 認識結果 → BERT」を融合して行う。評価は eval 23,444 区間、
フレーム単位 ＋ 停止規則（音声区間検出が発話末 +0.25 秒で発火する制約つき）。

| 指標 | 値 |
|---|---|
| 融合 macro-F1（最良・区間検出まで待つ） | **0.7537** |
| 音声のみ | 0.7354 |
| 天井（正解書き起こし） | 0.7839（残差 **0.0302**） |
| 出発点（2026-09 中旬） | 0.7429 |

**最良構成の実体**

| 要素 | パス |
|---|---|
| Stage1 ASR | `exp/asr_20260921-pureasr-blk42-sp`（CER 20.7 / WER 27.4、`block_size 42`） |
| Stage2 音声 | `exp/asr_20260924-turntaking-xfmr-pool_attn-frame2-blk42-same_n5_p30` の **`valid.loss.ave`** |
| フレーム確率 | `tt_preds/frame_attn-frame2-blk42_same_n5_{eval,train_dev}_valid.loss.ave.tsv` |
| テキスト | `exp/text_bert_asr`（**削除禁止**・再現不可、理由は §4） |
| 認識結果 | `exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave/{eval,train_dev_dec,train_nodup_dec}/text` |

**再現コマンド**

```bash
PY=/home/kobori/.conda/envs/espnet/bin/python
D=exp/asr_20260921-pureasr-blk42-sp/decode_cbs_transducer_bounded_asr_model_valid.loss.ave
$PY local/turntaking/frame_fusion.py \
    --stem frame_attn-frame2-blk42_same_n5 --bert exp/text_bert_asr \
    --asr_text $D/eval/text --dev_asr_text $D/train_dev_dec/text
```

---

## 2. 早期確定（早さ）の現状と、構造的な制約

| 規則 | macro-F1 | 短縮 |
|---|---|---|
| 区間検出まで待つ | 0.7537 | — |
| 完了τ=0.82 | 0.7529 | 0.172 秒 |
| 完了τ=0.70 | 0.7485 | 0.537 秒 |

**★重要**: 融合はテキストを**発話末以降のフレームにしか混ぜない**
（発話途中ではその発話の認識結果が存在しないため。因果性の制約）。したがって

- 「区間検出まで待つ」＝ 発話末以降 → テキストが効く
- **発話中に確定する早期確定は、音声モデル単独の質で決まる**

**早さを伸ばしたいなら、テキスト側を磨くのは無意味。音声側を良くするしかない。**

---

## 3. 次にやること（優先度順）

1. ~~**N-best 融合を新 Stage1 で取り直す。**~~ **2026-09-30 実施・効かず。**
   dev 選択（温度 2.0・音声 0.50）で eval **0.7532**（1-best 0.7537 に対し −0.0005）。
   dev では +0.0039 だったが eval に転移しない。オラクルの上限も +0.0069 → **+0.0050** に縮んだ。
   1-best が良くなった分、N-best の余地が消えた。`tt_analysis/nbest_fusion_blk42.txt`、
   実行は `tt_batch_logs/run_nbest_blk42.sh`（CPU 16 並列で eval＋dev 約 50 分）。
2. **【2026-09-30 15:25 から実行中】音声側 3 本を直列で。** `tt_batch_logs/run_audio_side.sh`
   （ログ `tt_batch_logs/audio_side.log`、段ごとの完了印 `tt_batch_logs/audio_side_*.done`）。
   A: 40ep 延長（`...frame2-blk42ep40...`、31ep から再開）→ B: Transformer 2 層ヘッド（`...blk42xh2...`）
   → C1: Stage1 blk60（`exp/asr_20260930-pureasr-blk60-sp`）＋デコード → C2: Stage2 blk60。
   各段とも ダンプ → `tt_analysis/frame_rules_{A,B,C2}.txt` → `tt_analysis/frame_fusion_{A,B,C2}.txt` まで自動。
   途中で落ちたら原因を直して同じスクリプトを cron で再投入すれば、完了印のある段は飛ばす。
   **A の結果（10/01 完了）**: 音声のみ 0.7354→**0.7391**（+0.0037）、融合（区間検出まで待つ）0.7537→0.7530（−0.0007）。
   早期確定は両方とも良くなった（音声のみ τ0.70: 0.7300/0.532 秒 → 0.7357/0.582 秒、
   融合 τ0.70: 0.7485/0.537 秒 → 0.7494/0.587 秒、融合 τ0.82: 0.7529/0.172 秒 → 0.7527/0.200 秒）。
   最良 valid loss_tag は 37ep の 0.540（30ep 時点 0.545）。`tt_analysis/frame_{rules,fusion}_A.txt`。
   **B の結果（10/03 完了）★新しい最良**: 音声のみ **0.7399**（+0.0045）、融合 **0.7575**（+0.0038、dev 選択 音声 0.60）。
   早期確定: 融合 τ0.70 で 0.7526/0.645 秒短縮（旧 0.7485/0.537 秒）、τ0.82 で 0.7568/0.225 秒（旧 0.7529/0.172 秒）。
   最良 valid loss_tag は 22ep の 0.534。`tt_analysis/frame_{rules,fusion}_B.txt`。
   → **B40（B を 40ep に延長）を B と C1 の間に差し込んだ**（10/03 18:42 開始）。C1 は 1ep 終了時点で止め、
   B40 の後に checkpoint から再開する。B40 の exp も Nepoch.pth はハードリンク（実増分 471 MB）。
   **B40 の結果（10/04 01:51 完了）**: 33ep で早期終了（最良 22ep から 11ep 更新なし、patience 10）。
   音声のみ 0.7396 / 融合 **0.7579**（τ0.70: 0.7532/0.637 秒、τ0.82: 0.7573/0.227 秒）。B とほぼ同じ（差 ±0.0004）。
   **延長の効果は B には無い**（A は最良が 27ep と遅く、延長中に更新が続いたので効いた）。
   **C1 の結果（10/05 20:37 完了、23ep で早期終了・最良 12ep）**: CER **20.8** / WER **27.7**
   （blk42-sp は 20.7 / 27.4）。valid loss は blk42 より約 0.8 低かったのに CER は改善せず＝**代理指標の裏切り 6 件目**。
   **C2 と D は中止（10/05 21:34）**：別作業 `docs/plan.md` ⓪ で、旧分割は評価話者の 70%（54/78 人）が学習と重なると判明し、
   話者が重ならない新分割（`train_g/dev_g/eval_g`）に作り直すことになった。`tt_batch_logs/run_gsplit_data.sh` が
   C1 終了後に C2 を止めてデータ準備を開始。D の待機スクリプトは本体停止を検知して正しく中止した。
   以降の実験は新分割で行う（旧分割の数値とは比べない）。
   **D（10/05 18:23 から待機していたが中止）**: `tt_batch_logs/run_audio_side_D.sh`（ログ `audio_side_D.log`）。
   本体の C2 評価完了を待ち、C2 が blk42 selfattn（音声のみ 0.7354 / 融合 0.7537）をどちらかで上回れば
   D = blk60 ＋ Transformer 2 層ヘッド（config `myconf/.gen_pool_attn-frame2-blk60xh2_same_n5_p30.yaml`）を学習・評価する。
   最後に C2/D の早期確定の規則を dev で選ぶ（`tt_analysis/commit_rules_C2_D_persist3.txt`）。
   Stage2 の config は `num_workers: 8`（既定 1 ではデータ待ちが 1 ステップの約 7 割。学習データは回転式 HDD 上）。
   **共用サーバなのでデータを SSD 等へ複製しないこと**（ユーザ指示）。
   A の exp は元の `...frame2-blk42...` の複製だが、`11〜30epoch.pth` は元とハードリンク（実増分 447 MB）。
   どちらを消しても他方は壊れない。上書きされる `checkpoint.pth` と `*.ave*.pth` は実体の複製。
   以下は実施前の記述。
   **音声側をさらに伸ばす（早さに直結）。**
   - `block_size` 42 → 60 前後（左文脈 1.19 → 1.9 秒）。ただし CEJC の平均発話長 1.5 秒なので飽和気味
   - ヘッドの変更（`docs/HANDOFF.md` §5.3、config だけで済む低コスト）
   - `max_epoch` 30 → 40 の延長（27ep 最良・30ep 横ばいだったので伸びは小さい見込み）
3. **【2026-10-05 実施】早期確定の規則を dev で選び直した** → `docs/commit_rules_20261005.md`。
   従来の「最良 τ」は eval で選んだ楽観値だった。単純な完了τを dev で選ぶと eval で条件を破りやすい（5/6 件）。
   **「余裕δ（p完了−p継続）＋持続 3 フレーム」を dev で選ぶと、融合では全モデルで eval でも条件を守る。**
   B 融合で δ0.56 持続3: 0.7566 / 0.226 秒短縮。規則の工夫で短縮はほぼ増えず、早さはモデルで決まる。
   以下は実施前の記述。
   **早期確定の規則を詰める。** 現在の最良規則は「完了クラスだけに閾値」
   （`docs/report_20260928_accuracy_speed.md` §6）。`local/turntaking/response_tradeoff.py` を
   融合確率に対応させ、「どちらの失敗も基準 +0.005 以内で最速」の規則を選び直す。
4. **【2026-09-30 実施】ラベル品質の検証** → `docs/label_quality_gold1000.md`。
   人手正解で測ると音声モデルの二値一致は 0.697（LLM ラベルで測ると 0.648）。LLM ラベル自体が人手と 0.742。
   以下は実施前の記述。
   **ラベル品質側の検証**（`docs/HANDOFF.md` §5.1）。天井 0.7839 との差 0.0302 を全部
   埋めても、その先には継続/完了の二値分離 0.65 という別の壁がある。
   LLM 付与ラベルの人手一致 F1 は 0.805。**時間対効果はこれが最大の可能性**。

---

### 2026-10-05 ディスクの片付け（ユーザ指示）

`exp/` を 448 GB → 87 GB に。基準は「これからの実験で使わないもの」。
- 消した：途中エポックの重み `Nepoch.pth`・`checkpoint.pth`（学習中の `asr_20260930-pureasr-blk60-sp` を除く全実験）と、
  リンク先が消えた `*.best.pth` / `latest.pth` のシンボリックリンク。一覧は `tt_batch_logs/cleanup_20261005_deleted.txt`。
- 残した：平均済みモデル `valid.*.ave*.pth`（リンク先の実体も）、config、ログ、デコード結果。平均モデルが無い古い 16 本は best の重みを残した。
- **以後、どの実験も checkpoint からの延長学習・別基準での再平均はできない**（延長が要るなら学習し直し）。

## 4. やらないこと（試して打ち止め・理由つき）

| やらないこと | 理由 |
|---|---|
| **テキスト側（BERT）の再学習・アンサンブル** | BERT 8 本＋2 アイデアで 1 つも 0.7537 を超えず。`docs/HANDOFF.md` §2.7 に全数値 |
| **`valid.tag_acc.ave` への差し替え** | 2026-09-30 実施。音声のみ 0.7354→0.7338、融合 0.7537→**0.7528** と悪化。ただし τ=0.70 では短縮が 0.537→0.572 秒に伸びる（精度は 0.7485→0.7469）ので、**早さ優先ならこちらも選択肢** |
| **旧 `1017-multitask-transducer`（CER 19.6）との比較** | 比較不能。出力がカタカナのモーラ列で参照が別物。`docs/stage1_asr_improvement.md` §4.5 |
| **過去発話を文脈として渡す** | 19 条件すべて「過去なし」を超えず。`docs/past_pooling_methods_examples.md` |
| **N-best の候補数を 5 → 10** | 実測が動かない。`docs/HANDOFF.md` §2.5 |
| **MBR など「候補を選ぶ」系** | 4 連敗。重み付けの改善方向で考えること |

### `exp/text_bert_asr` を消してはいけない理由

0.7537 を出す資産だが、**学習引数の記録が無く再現できない**。旧仮説で今のスクリプトで
再学習しても 0.7516 止まりで、差 +0.0021 の出所が不明。
`local/turntaking/text_ceiling.py` に**乱数シード制御が無い**ので同一条件の再現自体ができない。
→ 今後 BERT を学習するなら、まず seed 固定を入れてから回すこと。

---

## 5. このプロジェクト固有の落とし穴

### 代理指標が本番指標を裏切る（5 件確認済み）

| 代理指標 | 本番指標 | 結果 |
|---|---|---|
| valid loss（Stage1） | CER | 順位が逆転 |
| valid loss_tag（Stage2） | macro-F1 | 27ep で旧 39ep 最良を超えたが +0.0037 だけ |
| BERT 単体 macro-F1 | 融合後 macro-F1 | 0.7391→0.7468 と上がったのに融合は 0.7537→0.7496 |
| `valid.tag_acc`（タスクに近い選択） | 融合 macro-F1 | 0.7537→0.7528 と悪化 |
| 学習テキストの汚さ（帰属） | 対照実験 | 主因は別（記録なしのハイパラ差）だった |

**中間報告で代理指標だけを見て判断しないこと。最終判断は必ずデコード／融合まで通す。**

### 実行環境（silver11）

- **長時間ジョブは tmux やログインシェルから直接起動しない。** `systemd-oomd` が
  `user@1609.service` 配下の cgroup をメモリ圧で kill する。**一度きりの cron エントリ**で
  起動すると `/system.slice/cron.service` に入り監視対象外になる:

  ```bash
  echo "* * * * * /path/to/script.sh >> /path/to/log 2>&1" | crontab -
  # スクリプト先頭で crontab -r して一度きりにする
  ```

- **`sudo` は使えない。**
- **dump と GPU 学習を並行させない。** I/O 律速で `iter_time` が 100 倍悪化する。
- **監視・待機も cron に置くこと。** `nohup ... &` の子プロセスはセッション終了時に
  片付けられて消える。
- **追記式ログ（`>>`）を素朴に grep しない。** 過去の実行の「失敗」を拾って誤検知する。
  必ず最後の開始マーカー以降だけを見る。

### blk42 で Stage2 を回すときの設定

- `batch_bins` は半分（700000 → 350000）。`block_size` 42 は自己注意が (42/18)²≒5.4 倍で
  700000 では CUDA OOM。`accum_grad` を 8 → 16 にすれば実効バッチは旧と同じ 5,600,000。
- **`valid_batch_bins` は 300000 から下げてはいけない。** 下げると検証バッチが小さくなり、
  `past_speech` が 320 サンプル＝20 ms の無音スタブ（過去なしを表す。検証 14,039 件中 512 件）
  だけで構成されるバッチが生じ、conv2d subsampling が
  `Kernel size can't be greater than actual input size` で落ちる。**1 エポック目の検証で必ず落ちる。**
- 1 エポック約 3 時間、30 エポックで約 4 日。
- `run_xfmr_pool.sh` は **stage 10 決め打ち**。stats を作り直したくないときは asr.sh を
  `--stage 11` で直接呼ぶ（`tt_batch_logs/run_frame2_blk42_s11.sh` が実例）。

---

## 6. 読むもの

| ファイル | 内容 |
|---|---|
| `docs/HANDOFF.md` | **まずこれを全部。** §2.6 と §2.7、その後の「次にやること」 |
| `docs/stage1_asr_improvement.md` | Stage1 を CER 23.3→20.7 にした記録。寄与内訳と落とし穴 |
| `docs/report_20260928_accuracy_speed.md` | 精度と早さの発表用まとめ。停止規則の比較 |
| `docs/past_pooling_methods_examples.md` | 過去要約 7 手法の具体例つき説明 |
| `docs/peft_report.md` | LoRA / Adapter（旧 Stage1 前提） |
| `docs/past_context_pooling_report.md` | 過去発話 7 手法の詳細（旧 Stage1 前提） |
| `docs/label_ceiling_analysis.md` | ラベル品質の天井分析 |

**注**: `peft_report.md` と `past_context_pooling_report.md` の数値はすべて
旧 Stage1（`20260713-pureasr` / `block_size 18`）前提。冒頭に警告を入れてある。

### 実行スクリプトの実例

| 用途 | スクリプト |
|---|---|
| Stage1 学習（blk42 + sp） | `tt_batch_logs/run_sp_ext.sh` |
| Stage2 学習（blk42、stage 11 直行） | `tt_batch_logs/run_frame2_blk42_s11.sh` |
| ダンプ → 停止規則 → 融合 | `tt_batch_logs/run_phase4.sh` |
| train_nodup 再デコード → BERT 再学習 → 融合 | `tt_batch_logs/run_phase5.sh` |
| BERT アンサンブル比較 | `tt_batch_logs/run_step2_ens.sh`, `run_step3.sh` |
| チェックポイント差し替え | `tt_batch_logs/run_step4.sh` |
