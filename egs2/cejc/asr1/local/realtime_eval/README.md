# realtime_eval：実時間の評価

手順は [07 実時間の評価](../../docs/experiments/07_realtime_eval/README.md) と [08 別の対話データ](../../docs/experiments/08_other_corpora/README.md)．

silver13 の `~/2026/work/rt_eval_20261102/` で使ったもの（2026-11）．録音全体を頭から 32 ms ずつ，デモと同じ実時間のプログラム（`epd.engine`，別リポジトリ exp-realtime-epd）に流して採点する．

## 主なもの

- `rt_eval.py`：実際に流して応答の時刻を書き出す．`rt_score.py`：CEJC のラベルで採点する．
- `rt_dump.py`：応答はさせずに，各時刻の silero-vad の出力とスコアを書き出す．
- `rt_rule.py`：書き出したスコアに応答の決め方 (g1, th1, g2, th2, S, phi) を後から当てはめて採点する．`rt_pick.py`：開発データで決め方を選び，評価データに当てはめる．
- `rt_curve.py`：cut-in rate と latency の曲線．
- `kobori_rt_dump.sh`・`kobori_rt_real.sh`：書き出しと実際に流す確認を並列で回す．
- `kobori_chain*.sh`：実際に回したジョブ（日付と内容は各ファイルの頭）．
- `rt_silero.py`：silero-vad のみ（無音が一定時間続いたら応答）の決め方を選ぶ．`rt_table.py`：実際に流した結果を `rt_rule.py` と同じ数え方で採点する．
- `kobori_rt_dump10.sh`・`kobori_rt_real10.sh`・`run_dump10.sh`：silver10 で回したもの．`kobori_rt_dump_pool.sh`：exp-realtime-epd の複製で書き出す（全部書けたときだけ .done を置く）．

## 分析

- `rt_miss*.py`・`rt_misscat.py`・`rt_chunk.py`：応答しなかった区間の内訳．
- `rt_weak.py`・`rt_hes*.py`・`count_fd.py`・`rt_after_cont.py`：言いよどみ・発話の途中の間での応答．
- `rt_soma.py`・`rt_miss_soma.py`：waseda_soma（人が付けたターンのラベル）での採点．
- `rt_hazumi*.py`：Hazumi での採点．
- `miss_breakdown.py`・`seg_mismatch.py`・`late_sil.py`・`ev2tsv.py`：応答しなかった区間を，音声区間と書き起こしの区間のずれ方で分ける．
- `rt_pick_cap.py`・`kobori_cap.sh`：無音が一定の長さを超えた後のスコアを固定して選び直す．
- `cutin_dist.py`：cut-in に数えた区間ごとの，無音の長さと応答までの時間．
- `check_tailcount.py`・`check_incmean.py`：入力を伸ばしたとき後から値が変わるフレームの数と，フレームの平均を漸化式で更新しても今の計算と一致するか．

データ（CEJC・waseda_soma・Hazumi の音声と書き起こし）と重みは含まない．

## データの準備

- `soma_prep.py`・`soma_cejc_tag.py`：waseda_soma を CEJC と同じ形にする．
- `hazumi_extract.py`：Hazumi の動画からユーザの音声を取り出す．
