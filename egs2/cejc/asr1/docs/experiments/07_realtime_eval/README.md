# 07 実時間の評価

録音全体を頭から 32 ms ずつ，デモと同じ実時間のプログラム（exp-realtime-epd の `epd.engine`）に流し，応答した時刻を書き起こしのラベルで採点する．
発話の頭は silero-vad が声を拾った所で，計算の遅れも結果に入る．今の報告の表はすべてこの評価．

[← 全体の README](../../../../../../README.md)

- コード：`local/realtime_eval/`（silver13 では `~/2026/work/rt_eval_20261102/`．出力もここに出る）
- 実時間のプログラム：silver13 の `~/2026/work/exp-realtime-epd`（非公開．`weights/` に重みを置く）
- 環境：提案手法は conda 環境 `espnet`，ほかは `epd`．Easy Turn はさらに `easyturn` でサーバを立てる．

## 応答の決め方

無音（silero-vad の確率が 0.5 未満の続き）の間にスコアを見て，次のどれかを満たした最初の時刻で応答する．

| 記号 | 意味 |
| --- | --- |
| `g1`・`th1` | 無音が g1 秒続き，P(完了) ≥ th1 |
| `g2`・`th2` | 無音が g2 秒続き，P(完了) ≥ th2（g2 > g1，th2 < th1） |
| `S`・`phi` | 無音が S 秒続き，P(相槌) < phi |

まとめて `g1,th1,g2,th2,S,phi` と書く（例：提案手法 `0.4,0.84,2.0,0.4,3,0.1`）．

## 準備：silero-vad の確率（CEJC）

```bash
cd egs2/cejc/asr1
python local/turntaking/vad_silero_probs.py --dset dev_g --nj 4    # → tt_analysis/vad_silero/dev_g/<録音>.npy
python local/turntaking/vad_silero_probs.py --dset eval_g --nj 4
```

## 提案手法を評価する（スコアを書き出して，決め方を後から当てはめる）

提案手法は今の音声区間だけを入れるので，スコアは前に応答したかどうかによらない．そのため，スコアを 1 回書き出せば，約 1 万通りの決め方を後から試せる．

```bash
cd local/realtime_eval        # silver13 では ~/2026/work/rt_eval_20261102
S=<asr1>/tt_analysis/vad_silero

# 1. スコアの書き出し（GPU，6 並列）→ dump_<データ>_<名前>/
bash kobori_rt_dump.sh dev_g 6 espmk1 weights/encac_espmk1.pt
bash kobori_rt_dump.sh eval_g 6 espmk1 weights/encac_espmk1.pt

# 2. 開発データで決め方を選び，評価データに当てはめる
python rt_pick.py <asr1>/data/dev_g $S/dev_g dump_dev_g_espmk1 <asr1>/data/eval_g $S/eval_g dump_eval_g_espmk1

# 3. 選んだ決め方で評価データを実際に流し，後から当てはめた結果と合うかを確かめる
bash kobori_rt_real.sh eval_g 6 real_espmk1_p weights/encac_espmk1.pt --g1 0.4 --th1 0.84 --g2 2.0 --th2 0.4 --S 3 --phi 0.1
python rt_rule.py <asr1>/data/eval_g $S/eval_g dump_eval_g_espmk1 --rule 0.4,0.84,2.0,0.4,3,0.1 --check eval_g_real_espmk1_p
```

- `rt_pick.py` は，開発データで cut-in rate・応答しなかった割合・相槌の後に応答した割合の上限を決め，その中で latency が最も短い決め方を選ぶ（上限の組は中に書いてある．報告書は cut-in rate ≤ 6〜8%）．
- 1 つの決め方だけを採点するときは `rt_rule.py ... --rule g1,th1,g2,th2,S,phi`，全部を試すときは `--grid --cut 0.08 [--eval <評価 data> <silero> <書き出し>]`．
- cut-in rate と latency の曲線は `rt_curve.py`（引数は `rt_pick.py` と同じ）．

## 比較手法を評価する（決め方ごとに実際に流す）

エンコーダ固定＋Transformer と Easy Turn は，応答するたびに発話を区切り直すので，後から当てはめた結果が実際と合わない．決め方ごとに開発データを実際に流して選び，選んだもので評価データを流す．計算の遅れが結果に入るので，1 つずつ流す．

```bash
cd ~/2026/work/exp-realtime-epd
# エンコーダ固定＋Transformer（CPU 6 並列）
for i in 0 1 2 3 4 5; do
  python <realtime_eval>/rt_eval.py --wav_scp <asr1>/data/dev_g/wav.scp --out <realtime_eval>/dev_g_pg1x_g0.5_th0.79 \
      --shard $i --nshard 6 --model proposed_new --no_past --g1 0.5 --th1 0.79 --g2 2.0 --th2 0.4 --S 3 --phi 0.5 &
done; wait

# 採点
python <realtime_eval>/rt_score.py <asr1>/data/dev_g <realtime_eval>/dev_g_pg1x_g0.5_th0.79 $S/dev_g
```

- 決め方を動かして回したジョブ：`kobori_chain22.sh`（エンコーダ固定＋Transformer），`kobori_chain23.sh`（Easy Turn．`tools/et_server.py` を 4 つ立てて 4 並列）．
- silero-vad のみは `--model silero`．

## 指標

| 指標 | 数える区間 | 数え方 |
| --- | --- | --- |
| cut-in rate | 完了と継続 | 完了の区間で途中の応答があった，または継続の区間で途中か後の応答があった割合 |
| latency の中央値 | 完了のうち，次の区間まで 2 秒以上空いたもの | 区間末から後の応答までの時間．応答しなかった区間は無限に遅いとして含める |
| 応答しなかった | 同上 | 後の応答が無かった割合 |
| 相槌の後に応答 | 相槌 | 途中か後の応答があった割合 |
| 継続の後の間が 2〜3 秒で待てた | 継続のうち，次の区間までの間が 2〜3 秒のもの | 応答しなかった割合 |

silero-vad が一度も声を拾わなかった区間は数えない．方式どうしは cut-in rate をそろえて latency を比べる．

## 分析のスクリプト

| 調べたいこと | スクリプト |
| --- | --- |
| 応答しなかった区間の内訳 | `rt_miss.py`・`rt_misscat.py`・`rt_chunk.py`・`off_max.py` |
| どの条件（g1・g2・S）で応答したか | `rt_cond.py` |
| 発話の途中の間での応答 | `rt_weak.py`・`rt_hes.py`・`rt_hes_q.py`・`rt_cutin_examples.py` |
| 言いよどんだ後の latency | `rt_after_cont.py` |
| オフラインとの違い | `rt_vs_off.py`・`rt_diff.py`・`rt_head.py`・`rt_replay.py` |
| 入力の範囲（発話の頭から・直近 N 秒・今の音声区間だけ）を変える | `kobori_rt_ctx.sh` |

## 注意

- `rt_rule.py` を import するスクリプト（`rt_pick.py`・`rt_curve.py`・`rt_soma.py` など）は，同じ場所で動かす．
- 実時間の評価は録音ごとにメモリを使うので（1 本 3.5 GB まで），同じ機械の学習と並べるときは `guard.sh` で見張る．
