"""後から当てはめた応答を，rt_eval の出力と同じ形（録音ごとの tsv：t・speech_end・cond）で書き出す（miss_breakdown.py に渡す．11/20）
  python ev2tsv.py <data> <silero> <dump> <g1,th1,g2,th2,S,phi> <出力>"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from rt_rule import Data, OFF
data, sdir, ddir, rule, out = sys.argv[1:6]
r = tuple(None if v == "-" else float(v) for v in rule.split(","))
D = Data(data, sdir, ddir)
t, se, c = D.events(r)
os.makedirs(out, exist_ok=True)
k = np.floor(t / OFF).astype(int)
for i, rec in enumerate(D.recs):
    with open(f"{out}/{rec}.tsv", "w") as f:
        f.write("t\tspeech_end\tcond\n")
        for a, b, cc in zip(t[k == i] - i * OFF, se[k == i] - i * OFF, c[k == i]):
            f.write(f"{a:.4f}\t{b:.4f}\t{cc}\n")
