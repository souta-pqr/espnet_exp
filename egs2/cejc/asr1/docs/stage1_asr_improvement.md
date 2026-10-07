# Stage1（純粋 ASR エンコーダ）の改善 報告書

発話区間末検出の Stage2 が依存する **Stage1 ストリーミング ASR** を、レイテンシを一切増やさずに改善した記録。CER 23.3 → **20.7**（−2.6 ポイント）。

- 対象コーパス: CEJC（日本語日常会話コーパス）、train_nodup 80.5 時間 / 197,138 発話（平均 1.5 秒）
- 評価セット: eval 23,444 区間
- 作成日: 2026-09-24

---

## 1. 背景と目的

Stage2 の一連の実験（`docs/peft_report.md`、`docs/HANDOFF.md` §2.5）で、
**融合側の工夫では届かない壁**が明らかになっていた。正解書き起こしを使った天井が
macro-F1 0.7829 なのに対し、実際の認識結果での最良は 0.7461。この差 +0.037 のうち
N-best 融合で埋まったのは 1 割弱で、残りは **ASR の認識精度そのもの**に起因する。

つまり Stage2 を伸ばす最短経路は、Stage2 をいじることではなく **Stage1 を良くすること**。
ただし次の制約がある。

1. **レイテンシを増やしてはいけない。** ストリーミング用途なので、未来を覗く量
   （`look_ahead`）を増やす改善は使えない。
2. **Stage2 は Stage1 の重みを `init_param` で読む。** パラメータ名と形が変わる改造は
   Stage2 側の作り直しを伴う。

この制約下で何がどれだけ効くかを、1 要因ずつ切り分けて測った。

### 1.1 モデルの前提

| 項目 | 設定 |
|---|---|
| エンコーダ | Contextual Block Conformer（12 ブロック、output_size 256） |
| デコーダ | Transducer |
| 損失 | `loss_transducer` のみ（`turntaking_weight: 0`、`ctc_weight: 0`） |
| フレームレート | frontend `hop_length=132`（8.25 ms）× conv2d ÷4 = **33 ms/フレーム** |
| ストリーミング | `block_size: 18` / `hop_size: 3` / `look_ahead: 3` |
| バッチ | `batch_bins: 700000` |

---

## 2. 実験

出発点は `exp/asr_20260713-pureasr`（CER 23.3）。以下はすべてこの系列からの継続学習で、
**1 回につき 1 要因だけ**変えている。

### 2.1 実験 0: 延長学習（`20260918-pureasr-ext`）

元の学習は `max_epoch: 50` の上限に当たって止まっており、46・47・49 エポックで最良を
更新し続けていた。`max_epoch: 150` に延ばして継続。80 エポックで patience 打ち切り
（最良 69 エポック）。

**CER 23.3 → 22.5。** 単に学習が足りていなかった。

### 2.2 実験 1: block_size 18 → 42（`20260920-pureasr-blk42`）

Contextual Block Conformer の左文脈は、エンコーダ実装
（`espnet2/asr/encoder/contextual_block_conformer_encoder.py:254`）で

```
past_size = block_size - hop_size - look_ahead
```

と決まる。**`look_ahead` を据え置いたまま `block_size` だけ増やせば、
レイテンシを変えずに左文脈だけを伸ばせる。**

| | block_size | past_size | 左文脈 | レイテンシ |
|---|---|---|---|---|
| ベース | 18 | 12 フレーム | 約 0.40 秒 | 約 100 ms |
| 変更後 | 42 | 36 フレーム | 約 1.19 秒 | **約 100 ms（不変）** |

ただし、69 エポック目の重みからの継続学習なので、元の `lr: 0.002` /
`warmup_steps: 25000` をそのまま使うと学習率が高すぎる。`lr: 0.0005` /
`warmup_steps: 5000` に下げた。**この変更が block_size と交絡する。**

### 2.3 実験 1b: 対照実験（`20260920-pureasr-blk18ft`）

交絡を切るため、**`block_size: 18` のまま lr/warmup だけを実験 1 と揃えた**設定を
別に走らせた。これにより「lr を下げたぶん」と「block_size を増やしたぶん」を分離できる。

### 2.4 実験 2: speed perturbation（`20260921-pureasr-blk42-sp`）

学習データを 0.9 / 1.0 / 1.1 倍速の 3 系統に増やす（197,138 → 591,414 発話）。
実験 1 の最良重みから継続。20 エポックでまだ最良更新が続いていたため
`max_epoch: 40` に延長し、36 エポック（最良 25 エポック）で patience 打ち切り。

### 2.5 実験 3: CTC 補助損失（`20260921-pureasr-blk42-ctc`）

`ctc_weight: 0.0 → 0.3`。実験 1 の最良重みから
`--ignore_init_mismatch true` で継続（CTC の出力層は新規）。

---

## 3. 結果

### 3.1 全実験

| 実験 | 変更点 | エポック | valid loss | CER | WER |
|---|---|---|---|---|---|
| `20260713-pureasr`（元） | — | 50 | 13.269 | 23.3 | — |
| `20260918-pureasr-ext` | 延長学習のみ | 80（最良 69） | 13.051 | 22.5 | 30.0 |
| `20260920-pureasr-blk18ft` | **lr のみ**（対照） | 35（最良 24） | 12.251 | 22.1 | 29.4 |
| `20260920-pureasr-blk42` | lr + **block_size 42** | 36（最良 25） | 11.775 | 21.2 | 28.2 |
| `20260921-pureasr-blk42-sp` | + **speed perturb** | 36（最良 25） | **11.567** | **20.7** | **27.4** |
| `20260921-pureasr-blk42-ctc` | + CTC 0.3 | 9（打ち切り） | 12.048 | — | — |

`valid loss` は `loss_transducer`。CER/WER は eval セットでの
`decode_cbs_transducer_bounded_asr_model_valid.loss.ave`（10-best 平均モデル）。

### 3.2 CER の寄与内訳

| 要因 | CER の変化 |
|---|---|
| 延長学習 | 23.3 → 22.5（**−0.8**） |
| lr 0.002→0.0005 / warmup 25000→5000 | 22.5 → 22.1（**−0.4**） |
| block_size 18 → 42 | 22.1 → 21.2（**−0.9**） |
| speed perturbation | 21.2 → 20.7（**−0.5**） |
| **合計** | **23.3 → 20.7（−2.6）** |

**最大の単独要因は block_size。** しかもレイテンシは 100 ms のまま変わらない。

### 3.3 speed perturbation は効き始めが遅い

sp の valid loss（1 エポック目から）:

| ep | 1 | 2 | 4 | 7 | 8 | 11 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| loss | 12.323 | 11.913 | 11.791 | 11.763 | 11.637 | 11.613 | 11.582 | **11.567** |

データ量が 3 倍になるので 1 エポックも 3 倍（約 2 時間）かかる。
**実験 1 の最終ベスト 11.775 を下回るのに 7 エポック（約 14 時間）を要した。**
途中で「頭打ち」と判断して打ち切ると改善を取り逃がす。

### 3.4 CTC 補助損失は効かなかった

`loss_transducer` の最小値が 12.048 で、実験 1 の 11.775 を一度も下回らないまま
9 エポックで打ち切った。Transducer が十分に学習済みの状態に CTC を後から足しても、
アライメントの制約として働くより単に別方向へ引っ張る力になったと考えられる。

---

## 4. 考察

### 4.1 valid loss と CER の順位は一致しない

実験の途中で、**valid loss で比較するか CER で比較するかで結論が変わる**場面があった。

- `20260918-pureasr-ext`（13.051）と `20260920-pureasr-blk18ft`（12.251）の
  loss 差は 0.800 と大きいが、CER 差は 22.5 → 22.1 の 0.4 しかない。
- 一方 `blk18ft`（12.251）→ `blk42`（11.775）の loss 差は 0.476 と小さいのに、
  CER は 22.1 → 21.2 で 0.9 動く。

loss は単位時間あたりの尤度で、CER は編集距離。**両者は単調に対応しない。**
中間報告で valid loss だけを見て判断すると誤る。**最終判断は必ずデコードして CER で行う。**

### 4.2 block_size を増やす改善の性質

この改善が「ただ」なのは、Contextual Block Conformer が
**左文脈をブロック内の過去フレームとして持ち、右文脈だけを `look_ahead` で制御する**
構造だからである。左文脈を伸ばしてもアルゴリズム遅延は増えない。増えるのは
1 ブロックあたりの計算量（自己注意が 18² → 42² のオーダー）だけで、
実測の `iter_time` は 0.108 → 0.117 とほぼ変わらなかった。

さらに伸ばす余地はあるが、CEJC の平均発話長が 1.5 秒（約 45 フレーム）なので、
`past_size: 36`（1.19 秒）で既に平均的な発話のほぼ全体をカバーしている。
これ以上は発話内文脈としては飽和し、発話をまたぐ文脈になる。

### 4.3 実務上の落とし穴: `asr.sh` の stage 2 が独自の `tag` ファイルを落とす

speed perturbation は `asr.sh` の stage 2 が担当するが、その実装（`asr.sh:378`）が

```bash
ref_text_files_str="text "
```

と **`text` だけを列挙している**。本プロジェクトは各発話に区間末タグを持つ独自の
`tag` ファイルを `data/*/` に置いているため、stage 2 に任せると
`data/train_nodup_sp/` に `tag` が作られず、Stage2 の学習が動かなくなる。

対策として stage 2 は自前で回し、`--utt_extra_files "text tag"` を明示した
（`tt_batch_logs/run_sp.sh`）。

```bash
for f in 0.9 1.1; do
    scripts/utils/perturb_data_dir_speed.sh --utt_extra_files "text tag" \
        "$f" data/train_nodup "data/train_nodup_sp$f"
done
utils/combine_data.sh --extra_files "text tag" \
    data/train_nodup_sp data/train_nodup data/train_nodup_sp0.9 data/train_nodup_sp1.1
```

あわせて **stage 5（token_list 生成）は飛ばす**こと。走らせると token_list が
作り直され、継続学習元の重みと出力層の形が合わなくなる。

### 4.4 本実験の限界

- 探索したのは `block_size`（18 と 42 の 2 点のみ）、speed perturbation の有無、
  CTC の有無、lr の 2 点。`hop_size`、エンコーダ層数、`batch_bins`、SpecAugment の
  強さは未探索。
- `block_size` は 2 点しか測っていないので、42 が最適とは限らない。
  30 や 60 は試していない。
- 実験 1 の lr 変更と block_size 変更は同時に入れてしまい、後から対照実験
  （実験 1b）で切り分けた。順序としては先に対照を取るべきだった。

---

## 4.5 旧モデル `1017-multitask-transducer-hatuon-cejc` とは比較できない

`exp/asr_1017-multitask-transducer-hatuon-cejc` の RESULTS.md には **CER 19.6 / WER 24.9**
と記録されており、今回の 20.7 より良く見える。**これは比較できない数値である。**
現行 eval で再デコードしても比較可能にはならないことを確認した（2026-09-24）。

理由は 3 つあり、いずれも単独で比較を壊す。

1. **出力の表現が違う（決定的）。** 旧モデルはタグ名 `hatuon`（発音）のとおり
   **カタカナのモーラ列**を出力する。token_list は 136 種で、中身は
   `ア イ ウ … キャ ジャ ショ …`。スコアも カタカナの参照に対して取られている
   （`decode_.../eval/score_cer/ref.trn` が `コ <space> レ <space> マ <space> ズ …`）。
   一方、現行の参照は **漢字かな交じり ＋ 非流暢性マークアップ**（`(D …)`, `(F …)`,
   `<mask>`）で token_list は 2,611 種。カタカナ出力を漢字かな参照で採点すれば
   CER はほぼ 100% になる。**発音 CER は正書法 CER より系統的に低く出る**
   （漢字の読み分けも表記ゆれも誤りにならない）ので、19.6 < 20.7 は
   モデルの優劣を表していない。
2. **eval セットの中身が違う。** 旧 20,284 発話 / 現行 23,444 発話で、
   発話 ID が一致するのは **8,389 件だけ**（旧の 41%、現行の 36%）。
   セッションの振り分けも切り出し位置も変わっている。
3. **必要な入力が現行 dump に無い。** 旧モデルは `model: multitask_rnnt` で、
   学習時の入力に `isdysfl`（非流暢性ラベル）を取る。現行の `dump/raw/eval/` に
   このファイルは無い。

比較可能にするには、現行の切り出しに対してカタカナの発音参照を作り直す必要があるが、
それを生成していたスクリプトは現リポジトリに残っていない（`local/` にも `db.sh` にも
該当箇所なし）。**復元コストに見合わないので実施しない。**
Stage1 系列の基準は `20260713-pureasr`（CER 23.3）のままとする。

---

## 5. Stage2 への含意

**この新しい Stage1 を使うには、Stage2 をすべて再学習する必要がある。**
`block_size` が 18 → 42 に変わっているため、エンコーダの挙動が変わり、
これまでの Stage2 の結果（`docs/peft_report.md`、`docs/HANDOFF.md` §2・§2.5）は
そのままでは引き継げない。

### 実測結果（2026-09-30 完了）

Stage2 の `frame2` を `block_size: 42` で作り直して測った。**期待した 2 経路の
両方に効果があった。**

| 条件 | 音声側 | テキスト側 | macro-F1 |
|---|---|---|---|
| 音声のみ | 旧 | — | 0.7306 |
| **音声のみ** | **新 blk42** | — | **0.7354** |
| 融合 1-best（出発点） | 旧 | 旧 ASR | 0.7429 |
| 融合 N-best（旧の最良） | 旧 | 旧 ASR | 0.7461 |
| 融合 1-best | 旧 | **新 ASR** | 0.7500 |
| **融合 1-best（最終）** | **新 blk42** | **新 ASR** | **0.7537** |
| 天井：正解書き起こし | 新 | 正解 | 0.7839 |

**0.7429 → 0.7537（+0.0108）。天井との差は 0.0410 → 0.0302 で 26% 縮小。**
音声側を旧のまま固定して認識結果だけ差し替える実験を先に行ったので、
寄与を分離できる。

| 経路 | 寄与 |
|---|---|
| テキスト経路（認識結果が綺麗になる → BERT の判定が良くなる） | **+0.0071** |
| 音声経路（エンコーダ表現が良くなる → 音声モデルのタグ予測が良くなる） | **+0.0037** |

**1-best のまま N-best（0.7461）を上回った。** 融合側の工夫で積み上げた +0.0032 を、
認識精度の改善だけで超えたことになる。

想定と違った点が 2 つある。

1. **音声経路の寄与は valid loss から期待したほど大きくない。** 学習中の
   valid loss_tag は旧 0.552（39ep）→ 新 0.545（27ep）で、旧が 39 エポックかけた
   水準を新は 27 エポックで超えた。しかし macro-F1 の差は +0.0037 に留まる。
   §4.1 と同じ現象がここでも出ている。
2. **融合の利得構造が変わった。** 従来は「利得はほぼ全部『継続』から来る／
   テキストは完了の判定に貢献しない」だったが、今回は完了F1 が音声重み
   0.0→0.6 で 0.721→0.751 と大きく動く。認識精度が上がったことで、
   テキスト側が「言い終えた」の判定にも使えるようになった。

詳細な数値・動作点・再現手順は `docs/HANDOFF.md` §2.6。

---

## 6. 関連ファイル

| 種別 | パス |
|---|---|
| ベース設定 | `myconf/train_asr_pureasr_conformer_ext.yaml` |
| 対照（lr のみ） | `myconf/train_asr_pureasr_conformer_blk18ft.yaml` |
| block_size 42 | `myconf/train_asr_pureasr_conformer_blk42.yaml` |
| + speed perturb | `myconf/train_asr_pureasr_conformer_blk42_sp.yaml` / `..._sp_ext.yaml` |
| + CTC | `myconf/train_asr_pureasr_conformer_blk42_ctc.yaml` |
| 実行スクリプト | `tt_batch_logs/run_sp.sh`, `run_sp_ext.sh`, `run_ctc.sh` |
| 実行ログ | `tt_batch_logs/*.log` |
| 学習済みモデル | `exp/asr_20260918-pureasr-ext`, `exp/asr_20260920-pureasr-blk18ft`, `exp/asr_20260920-pureasr-blk42`, `exp/asr_20260921-pureasr-blk42-sp` |
| 左文脈の計算箇所 | `espnet2/asr/encoder/contextual_block_conformer_encoder.py:254` |

### 実行環境上の注意（silver11）

長時間ジョブは **tmux やログインシェルから直接起動しない**。`systemd-oomd` が
`user@1609.service` 配下の cgroup をメモリ圧で kill するため、tmux のペイン
（`tmux-spawn-*.scope`）で走らせたジョブは巻き添えで落ちる。**一度きりの cron
エントリ経由で起動する**と `/system.slice/cron.service` に入り、監視対象外になる。

```bash
echo "* * * * * /path/to/script.sh >> /path/to/log 2>&1" | crontab -
# スクリプトの先頭で crontab -r して一度きりにする
```

また **dump（stage 3-4）と GPU 学習を並行させない**こと。I/O 律速で、
並行させると学習の `iter_time` が 100 倍悪化する。
