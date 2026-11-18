#!/usr/bin/env python
"""話を取った継続の区間を，発話の頭と過去の文脈を替えて提案手法に流し直す（2026-11-03）．
  A：実時間と同じ（頭＝実時間の発話の頭，過去＝実時間の発話）．再現の確かめ
  B：頭だけ正解の区間の頭にする（過去は実時間の発話）
  C：頭も過去も正解の区間（オフラインと同じ作り方）
応答の決め方は proposed（無音 0.6 秒以上かつ P(完了) ≥ 0.84）．区間末から次の自分の区間の頭までに応答したら「取った」．
  cd ~/2026/work/exp-realtime-epd && python <ここ>/rt_replay.py cutin_proposed.tsv off_rows.tsv eval_g <data> out.tsv
"""
import csv, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epd.vad import SileroVAD, HOP, SR
from epd.models.proposed import ProposedScorer
from rt_eval import cap

cut, off, rt, data, out = sys.argv[1:6]
torch.set_num_threads(2)
offm = {(r["rec"], r["b"]): float(r["off_max"]) for r in csv.DictReader(open(off), delimiter="\t")}
wav = dict(l.split() for l in open(f"{data}/wav.scp"))
segs = {}
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); segs.setdefault(rec, []).append((float(b), float(e)))
sc = ProposedScorer("weights/proposed_gt.pt", async_past=False)
cap(sc)
G1, TH = 0.6, 0.84


def run(x, head, past, e, nxt):
    """head から nxt まで流し，[e, nxt) で応答したら (区間末からの秒, スコア) を返す"""
    sc.past = []
    for p in past:
        sc.add_past(p)                       # 直前 5 発話・30 秒までに切るのは add_past が行う
    if not past:
        sc.pv_ready = sc._past_vector()
    sc.start()
    vad = SileroVAD()
    i0 = max(0, int(head * SR) - 2 * SR) // HOP * HOP     # silero は 2 秒前から回しておく
    h = int(head * SR) // HOP * HOP
    for i in range(i0, h, HOP):
        vad(x[i:i + HOP])
    score = None
    first = True
    for i in range(h, int(nxt * SR) - HOP + 1, HOP):
        c = x[i:i + HOP]
        p, r = vad(c)
        for _, s, _ in sc.push(c, silent=p < 0.5):
            score = s
        t = (i + HOP) / SR
        if t >= e and r >= G1 and score is not None and score >= TH:
            return t - e, score
    return None, None


rows = list(csv.DictReader(open(cut), delimiter="\t"))
rows = [r for r in rows if r["間の他の人"] == "誰も話さない"]
SH, NSH = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
rows = rows[SH::NSH]
X = {}
with open(out, "w") as f:
    f.write("rec\tb\te\t間\toffline\trt_head\tA\tB\tC\n")
    for k, r in enumerate(rows):
        rec, b, e = r["rec"], float(r["b"]), float(r["e"]); nxt = e + float(r["間"]); t = e + float(r["区間末から応答まで"])
        if rec not in X:
            X = {rec: sf.read(wav[rec], dtype="float32")[0]}
        x = X[rec]
        ev = [y for y in csv.DictReader(open(f"{rt}/{rec}.tsv"), delimiter="\t") if y["model"] == "proposed"]
        cur = [y for y in ev if abs(float(y["t"]) - t) < 0.02][0]
        us = float(cur["utt_start"])
        rtpast = [x[int(float(y["utt_start"]) * SR):int(float(y["speech_end"]) * SR)] for y in ev if float(y["t"]) <= us][-5:]
        gpast = [x[int(bb * SR):int(ee * SR)] for bb, ee in segs[rec] if ee <= b][-5:]
        res = [run(x, us, rtpast, e, nxt), run(x, b - 0.2, rtpast, e, nxt), run(x, b - 0.2, gpast, e, nxt)]
        fmt = lambda v: "待った" if v[0] is None else f"{v[0]:.2f}秒 {v[1]:.2f}"
        f.write(f"{rec}\t{b:.2f}\t{e:.2f}\t{r['間']}\t{offm[(rec, r['b'])]:.2f}\t{b - us:.2f}\t" + "\t".join(fmt(v) for v in res) + "\n")
        f.flush()
        print(k, flush=True)
