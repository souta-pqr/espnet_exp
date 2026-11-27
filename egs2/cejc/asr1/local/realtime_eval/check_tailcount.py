import sys, wave, numpy as np, torch
sys.path.insert(0, ".")
from epd.models.encac import load_model
torch.set_num_threads(4)
m = load_model(sys.argv[1], "cpu")
def enc(a):
    t = torch.from_numpy(a)[None]
    f, fl = m.frontend(t, torch.tensor([t.size(1)])); f, fl = m.normalize(f, fl)
    h, hl, _ = m.encoder(f, fl)
    return h[0, :hl[0]]
res = {}
for rec, off in (("K008/K008_001/K008_001a_IC01", 20), ("C001/C001_001/C001_001_IC01", 60), ("K008/K008_001/K008_001a_IC02", 100)):
    w = wave.open(f"/autofs/diamond2/share/corpus/CEJC_safia/data/{rec}.wav"); w.setpos(16000 * off)
    x = np.frombuffer(w.readframes(16000 * 13), dtype=np.int16).astype(np.float32) / 32768
    with torch.no_grad():
        for L in (0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 3.0, 5.0, 8.0, 10.0):
            n = int(L * 16000); a = enc(x[:n])
            ch = []
            for add in (0.1, 0.5, 3.0):
                b = enc(x[:n + int(add * 16000)])
                d = (a - b[:len(a)]).abs().max(1).values
                bad = torch.nonzero(d > 1e-4).flatten()
                ch.append(len(a) - int(bad[0]) if len(bad) else 0)
            res.setdefault(L, []).append((len(a), ch))
print("入力(秒)\tフレーム数\t変わった末尾のフレーム数（+0.1 秒・+0.5 秒・+3 秒．録音 3 つ）")
for L, v in res.items():
    print(f"{L}\t{v[0][0]}\t" + "  ".join(str(c) for _, c in v))
