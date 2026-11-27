"""補助タスクの出力層（CTC．train_ac.py の best_aux.pt）で音声認識をして CER を測る（2026-11-26）．
入力は書き起こしの区間（頭から終わりまで），出力は CTC の貪欲法（各フレームで最も確率の高い文字．続く同じ文字と空白を消す）．
正解と認識結果は train_et_data.norm と同じ形に直す（タグを消し，<unclear>・<mask> は＊）．
  cd ~/2026/work/encac_pool
  ENC=... PYTHONPATH=~/2025/espnet python cer_aux.py <exp ディレクトリ（best.pt・best_aux.pt）> <POOL> <data> <出力の認識結果>
"""
import os, sys, wave
import numpy as np, torch, yaml, editdistance
sys.path.insert(0, os.getcwd())
import enc_plug as P
from train_et_data import norm, SR

ed, pool, data, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
P.POOL = pool
TOK = yaml.safe_load(open(P.GTCONF))["token_list"]
m = P.ESP(); m.load_state_dict(torch.load(f"{ed}/best.pt", map_location="cpu")); m = m.cuda().eval()
aux = torch.nn.Linear(m.head.in_features, len(TOK)); aux.load_state_dict(torch.load(f"{ed}/best_aux.pt", map_location="cpu")); aux = aux.cuda().eval()
wav = dict(l.split() for l in open(f"{data}/wav.scp"))
txt = dict((l.rstrip("\n").split(" ", 1) + [""])[:2] for l in open(f"{data}/text"))
segs = [l.split() for l in open(f"{data}/segments")]
err = n = 0
BS = 16
with torch.no_grad(), open(out, "w") as fo:
    for i in range(0, len(segs), BS):
        xs, us = [], []
        for u, rec, b, e in segs[i:i + BS]:
            w = wave.open(wav[rec]); s0 = int(round(float(b) * SR)); s1 = int(round(float(e) * SR))
            w.setpos(s0); xs.append(torch.from_numpy(np.frombuffer(w.readframes(max(s1 - s0, 400)), dtype=np.int16).astype(np.float32) / 32768)); us.append(u)
        dev = "cuda"
        lens = torch.tensor([len(x) for x in xs], device=dev)
        x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(dev)
        f, fl = m.frontend(x, lens); f, fl = m.normalize(f, fl)
        h, hl, _ = m.encoder(f, fl)
        best = aux(h).argmax(-1).cpu().numpy()
        for k, u in enumerate(us):
            s = best[k, :int(hl[k])]
            hyp = "".join(TOK[t] for j, t in enumerate(s) if t != 0 and (j == 0 or t != s[j - 1]) and not TOK[t].startswith("<"))
            ref = norm(txt[u])
            err += editdistance.eval(ref, hyp); n += len(ref)
            fo.write(f"{u}\t{ref}\t{hyp}\n")
print(f"{os.path.basename(ed)}  {data}  {len(segs)} 区間  正解 {n} 文字  CER {100 * err / n:.1f}%")
