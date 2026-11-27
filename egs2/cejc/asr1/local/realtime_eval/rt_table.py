"""実際に流した結果（rt_eval の出力）を，rt_rule.py と同じ数え方で採点する（比較手法の表を提案手法とそろえる．2026-11-23）．
後から当てはめた決め方も並べられる（決め方は g1,th1,g2,th2,S,phi．"-" は使わない）．
  python rt_table.py <data> <silero> <書き出し（区間と録音の一覧に使う）> <名前=rt_eval の出力 | 名前=@決め方> ...
"""
import os, sys
sys.path.insert(0, os.getcwd())
from rt_rule import Data, real_events

data, sdir, ddir = sys.argv[1:4]
D = Data(data, sdir, ddir)
print("方式\tcut-in rate\tlatency の中央値\t応答しなかった\t相槌の後に応答\t間 2〜3 秒で待てた\t応答した分の latency")
for x in sys.argv[4:]:
    name, v = x.split("=", 1)
    ev = D.events(tuple(None if s == "-" else float(s) for s in v[1:].split(","))) if v.startswith("@") else real_events(D, v)
    m = D.score(ev)
    print(f"{name}\t{m['cut']:.1%}\t{m['lat']:.2f}\t{m['miss']:.1%}\t{m['bc']:.1%}\t{m['wait']:.1%}\t{m['lat_ok']:.2f}")
