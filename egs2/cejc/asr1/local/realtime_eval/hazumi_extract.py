"""Hazumi の動画からユーザだけの音声を作る（2026-11-16．Hazumi1911 と同じ作り方）．16 kHz モノラルにし，システムの発話の時間と最初のやりとり以前を無音にする．
  python hazumi_extract.py 1712 1902"""
import csv, glob, os, subprocess, sys, wave
import numpy as np
H = "/autofs/diamond2/share/corpus/Hazumi"; O = "/autofs/diamond5/share/users/kobori/hazumi_audio"
for v in sys.argv[1:]:
    out = f"{O}/Hazumi{v}_user"; os.makedirs(out, mode=0o700, exist_ok=True)
    scp = open(f"{O}/Hazumi{v}_user.scp", "w"); tsv = open(f"{O}/Hazumi{v}_exchanges.tsv", "w")
    tsv.write("rec\tsys_start\tsys_end\tex_end\tnbar\tfiller\tnchar\n")
    for c in sorted(glob.glob(f"{H}/git/Hazumi{v}/dumpfiles/*.csv")):
        rec = os.path.basename(c)[:-4]; mp4 = f"{H}/extracted/Hazumi{v}_video/video/{rec}.mp4"
        if not os.path.exists(mp4): print("動画なし", rec); continue
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
        x = np.frombuffer(raw, np.int16).copy()
        rows = [(float(r["start(exchange)[ms]"]) / 1000, float(r["end(system)[ms]"]) / 1000, float(r["end(exchange)[ms]"]) / 1000) for r in csv.DictReader(open(c))]
        e = lambda a, b: float(np.sqrt(np.mean(x[int(a * 16000):int(b * 16000)].astype(np.float64) ** 2) + 1e-9))
        es = np.median([e(a, b) for a, b, _ in rows if b > a]); eu = np.median([e(b, c2) for _, b, c2 in rows if c2 > b])
        x[:int(rows[0][0] * 16000)] = 0
        for a, b, c2 in rows:
            x[int(a * 16000):int(b * 16000)] = 0
            tsv.write(f"{rec}\t{a}\t{b}\t{c2}\t0\t0\t0\n")
        w = wave.open(f"{out}/{rec}.wav", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes()); w.close()
        scp.write(f"{rec} {out}/{rec}.wav\n")
        print(rec, len(rows), f"システムの時間の音量 {es:.0f}・ユーザの時間 {eu:.0f}", flush=True)
