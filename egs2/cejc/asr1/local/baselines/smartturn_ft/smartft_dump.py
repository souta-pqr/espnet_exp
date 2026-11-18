"""CEJC で学習し直した Smart Turn（train_st.py の best.pth．2026-10-29）で，各区間の頭から区間末 +5 秒まで 0.1 秒ごとに「言い終えた確率」を出す（2026-10-21）。

入力は公式の inference.py と同じ：その区間（ターン）の頭から今までの本人の音声．8 秒より長ければ最後の 8 秒，
短ければ頭を 0 で埋めて 8 秒にし，WhisperFeatureExtractor(chunk_length=8, do_normalize=True) を通す。
時刻 t の出力は t までの音声だけで出る（先読みなし）。
出力は eot_eval.py の --dump 形式（utt true pos elapsed p_cont p_end p_bc）．p_end＝確率，p_bc＝0（相槌の確率は無い）。

公開の重みで作った ../baselines_20261021/smart_dump.py と同じ手順．違いは重みと，ONNX ではなく PyTorch（bf16）で動かすことだけ。
  python smartft_dump.py data/eval_g out/smartft_eval_g.part0.tsv 0 4 [最大の録音数]   # 録音を 4 つに分けた 0 番目
"""
import sys, os, wave, collections, time
import numpy as np, torch
from transformers import WhisperFeatureExtractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_model import SmartTurnV3

STEP, POST, SR, N8, BS = 0.1, 5.0, 16000, 8 * 16000, 64
data, out, part, nparts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
maxrec = int(sys.argv[5]) if len(sys.argv) > 5 else 0
fe = WhisperFeatureExtractor(chunk_length=8)
model = SmartTurnV3()
model.load_state_dict(torch.load("/home/kobori/2025/espnet/egs2/cejc/asr1/exp/smartturn_ft_20261028/best.pth", map_location="cpu"))
dev = torch.device("cuda")
model.to(dev).eval()
wavs = dict(l.split() for l in open(f"{data}/wav.scp"))
segs = collections.defaultdict(list)
for l in open(f"{data}/segments"):
    u, rec, b, e = l.split(); segs[rec].append((u, float(b), float(e)))
recs = sorted(segs)[part::nparts]
if maxrec:
    recs = recs[:maxrec]


def read(path):
    w = wave.open(path); assert w.getframerate() == SR and w.getnchannels() == 1
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def run(clips):
    f = fe(clips, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=N8,
           truncation=True, do_normalize=True).input_features.astype(np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.sigmoid(model(torch.from_numpy(f).to(dev)).float()).cpu().numpy().reshape(-1)


t0 = time.time()
with open(out, "w") as fo:
    if part == 0:
        fo.write("utt\ttrue\tpos\telapsed\tp_cont\tp_end\tp_bc\n")
    for i, r in enumerate(recs):
        x = read(wavs[r]); dur = len(x) / SR
        jobs = []                                    # (utt, 番号, 区間の頭からの時刻, 切り出しの始まり, 終わり)．音声は使う直前に切り出す
        for u, b, e in segs[r]:
            ts = np.arange(1, int(round((e + POST - b) / STEP)) + 1) * STEP
            ts = ts[b + ts <= dur]
            for j, t in enumerate(ts):
                s1 = int(round((b + t) * SR)); s0 = max(int(round(b * SR)), s1 - N8)
                jobs.append((u, j, t, s0, s1))
        for k in range(0, len(jobs), BS):
            bt = jobs[k:k + BS]
            p = run([np.pad(x[s0:s1], (N8 - (s1 - s0), 0)) for *_, s0, s1 in bt])
            for (u, j, t, _, _), pp in zip(bt, p):
                fo.write(f"{u}\t-1\t{j}\t{t:.4f}\t{1 - pp:.6f}\t{pp:.6f}\t0\n")
        print(f"{i + 1}/{len(recs)} {r} {len(jobs)} 件 {time.time() - t0:.0f} 秒", flush=True)
print("ok")
