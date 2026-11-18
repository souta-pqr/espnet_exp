# realtime_eval：実時間の評価

silver13 の `~/2026/work/rt_eval_20261102/` で使ったもの（2026-11）．録音全体を頭から 32 ms ずつ，デモと同じ実時間のプログラム（`epd.engine`，別リポジトリ exp-realtime-epd）に流して採点する．

## 主なもの

- `rt_eval.py`：実際に流して応答の時刻を書き出す．`rt_score.py`：CEJC のラベルで採点する．
- `rt_dump.py`：応答はさせずに，各時刻の silero-vad の出力とスコアを書き出す．
- `rt_rule.py`：書き出したスコアに応答の決め方 (g1, th1, g2, th2, S, phi) を後から当てはめて採点する．`rt_pick.py`：開発データで決め方を選び，評価データに当てはめる．
- `rt_curve.py`：cut-in rate と latency の曲線．
- `kobori_rt_dump.sh`・`kobori_rt_real.sh`：書き出しと実際に流す確認を並列で回す．
- `kobori_chain*.sh`：実際に回したジョブ（日付と内容は各ファイルの頭）．

## 分析

- `rt_miss*.py`・`rt_misscat.py`・`rt_chunk.py`：応答しなかった区間の内訳．
- `rt_weak.py`・`rt_hes*.py`・`count_fd.py`・`rt_after_cont.py`：言いよどみ・発話の途中の間での応答．
- `rt_soma.py`・`rt_miss_soma.py`：waseda_soma（人が付けたターンのラベル）での採点．
- `rt_hazumi*.py`：Hazumi での採点．

データ（CEJC・waseda_soma・Hazumi の音声と書き起こし）と重みは含まない．
