"""Smart Turn v3.2 を CEJC で学習し直す（公開の重みから続けて学習．2026-10-28，silver13）。

従来手法との比較を公平にするため，提案法（gt）と同じデータ・同じラベル・同じ読み出し位置で学習する。
  - データ：学習 data/train_gx，検証 data/dev_g（提案法と同じ）
  - ラベル：gemma のラベル（data/*/tag：0=継続，1=完了，2=相槌）．Smart Turn は 2 値なので 完了＝1，継続・相槌＝0
  - 読み出し位置：区間末の −0.6 秒〜+2.0 秒から一様に 1 つ（提案法 gt の tt_frame_win_sec 0.6・tt_tail_sec 2.0 と同じ）
  - 入力：公式の使い方と同じく，区間の頭から読み出し位置までの本人の音声（8 秒より長ければ最後の 8 秒，短ければ頭を 0 で埋める）
学習の設定は Smart Turn の train.py と同じ（lr 5e-5，4 エポック，バッチ 384，warmup 0.2，cosine，weight decay 0.01，
バッチごとの pos_weight = 負例 / 正例（0.1〜10））．バッチ 384 は 128 × 勾配の蓄積 3 で作る。
検証は区間ごとに位置を固定（seed 0）して，F1 が最良の重みを残す（train.py と同じく F1 で選ぶ）。

  python train_st.py <出力ディレクトリ> [--max_steps N]（N は速さとメモリを測るときだけ）
"""
import sys, os, math, time, wave, argparse
import numpy as np, torch, torch.nn as nn
from transformers import WhisperFeatureExtractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from st_model import load_from_onnx

ASR1 = "/home/kobori/2025/espnet/egs2/cejc/asr1"
ONNX = "/home/kobori/tt_base/smart_turn/smart-turn-v3.2-gpu.onnx"
SR, N8, PRE, POST = 16000, 8 * 16000, 0.6, 2.0
LR, EPOCHS, BS, ACC, WARM, WD = 5e-5, 4, 128, 3, 0.2, 0.01

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)


def load(d):
    wav = dict(l.split() for l in open(f"{ASR1}/data/{d}/wav.scp"))
    tag = dict(l.split() for l in open(f"{ASR1}/data/{d}/tag"))
    items = []
    for l in open(f"{ASR1}/data/{d}/segments"):
        u, rec, b, e = l.split()
        items.append((wav[rec], float(b), float(e), 1.0 if tag[u] == "1" else 0.0))
    return items


class DS(torch.utils.data.Dataset):
    def __init__(self, items, fixed):
        self.items, self.fixed = items, fixed
        self.fe = WhisperFeatureExtractor(chunk_length=8)
        self.off = np.random.default_rng(0).uniform(-PRE, POST, len(items)) if fixed else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, b, e, y = self.items[i]
        off = self.off[i] if self.fixed else np.random.uniform(-PRE, POST)
        w = wave.open(path)
        n = w.getnframes()
        s1 = min(int(round((e + off) * SR)), n)
        s0 = max(int(round(b * SR)), s1 - N8)
        s1 = max(s1, s0 + 1)
        w.setpos(s0)
        x = np.frombuffer(w.readframes(s1 - s0), dtype=np.int16).astype(np.float32) / 32768.0
        w.close()
        x = np.pad(x, (N8 - len(x), 0))
        f = self.fe(x, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=N8,
                    truncation=True, do_normalize=True).input_features[0].astype(np.float32)
        return torch.from_numpy(f), torch.tensor(y)


def loader(items, fixed, shuffle):
    return torch.utils.data.DataLoader(DS(items, fixed), batch_size=BS, shuffle=shuffle, num_workers=args.workers,
                                       pin_memory=True, drop_last=shuffle, persistent_workers=True, prefetch_factor=4)


tr, dv = load("train_gx"), load("dev_g")
print(f"学習 {len(tr)} 区間（完了 {sum(i[3] for i in tr):.0f}）．検証 {len(dv)} 区間（完了 {sum(i[3] for i in dv):.0f}）", flush=True)
dl_tr, dl_dv = loader(tr, False, True), loader(dv, True, False)

model, miss, unexp = load_from_onnx(ONNX)
assert not miss and not unexp
dev = torch.device("cuda")
model.to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
total = EPOCHS * (len(dl_tr) // ACC)
if args.max_steps:
    total = args.max_steps
warm = int(WARM * total)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))


def evaluate():
    model.eval()
    ps, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for f, y in dl_dv:
            ps.append(torch.sigmoid(model(f.to(dev, non_blocking=True)).float()).cpu())
            ys.append(y)
    model.train()
    p, y = torch.cat(ps).numpy(), torch.cat(ys).numpy()
    pr = p >= 0.5
    tp, fp, fn = (pr & (y == 1)).sum(), (pr & (y == 0)).sum(), (~pr & (y == 1)).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    bce = float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))
    return f1, bce, float((pr == (y == 1)).mean())


if not args.max_steps:                                       # 速さを測るときは検証を飛ばす
    f1, bce, acc = evaluate()
    print(f"[学習前（公開の重み）] 検証 F1 {f1:.4f}  BCE {bce:.4f}  正解率 {acc:.4f}", flush=True)
best = -1.0
step, t0, run = 0, time.time(), []
model.train()
eval_every = max(1, (len(dl_tr) // ACC) // 4)                 # 1 エポックに 4 回
done = False
for ep in range(EPOCHS):
    for k, (f, y) in enumerate(dl_tr):
        f, y = f.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        pos = y.sum().clamp(min=1)
        pw = ((len(y) - y.sum()) / pos).clamp(0.1, 10.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(f).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pw) / ACC
        loss.backward()
        run.append(loss.item() * ACC)
        if (k + 1) % ACC:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 100 == 0 or (args.max_steps and step % 10 == 0):
            print(f"エポック {ep + 1} 更新 {step}/{total}  損失 {np.mean(run[-300:]):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time() - t0:.0f} 秒  GPU メモリ最大 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        if not args.max_steps and (step % eval_every == 0 or step == total):
            f1, bce, acc = evaluate()
            mark = ""
            if f1 > best:
                best = f1
                torch.save(model.state_dict(), f"{args.out}/best.pth")
                mark = "  ← 最良"
            print(f"[更新 {step}] 検証 F1 {f1:.4f}  BCE {bce:.4f}  正解率 {acc:.4f}{mark}", flush=True)
        if step >= total:
            done = True
            break
    if done:
        break
torch.save(model.state_dict(), f"{args.out}/last.pth")
print(f"完了．最良の検証 F1 {best:.4f}  {time.time() - t0:.0f} 秒", flush=True)
