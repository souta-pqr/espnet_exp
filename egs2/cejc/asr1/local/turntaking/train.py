#!/usr/bin/env python
"""turn-taking ヘッドの学習（凍結エンコーダ特徴シャード上）.

損失 = 時間重み付き分類CE(turn-end/相槌/継続) + λ·未来VA予測BCE(自己教師).
  時間重み w(t): 序盤フレームは軽く罰し、終盤は重く（早すぎ誤判定を抑え遅延を罰する近似）。
学習対象はヘッドのみ。エンコーダ/ASR は触らない。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from model import TurnTakingHead


def read_manifest(d):
    d = Path(d)
    return [d / ln.strip() for ln in open(d / "manifest.txt") if ln.strip()]


def split_shard(npz):
    z = np.load(npz)
    feats, va, lens = z["feats"], z["va"], z["lengths"]
    labs = z["labels"]
    off = np.concatenate([[0], np.cumsum(lens)])
    items = []
    for i in range(len(lens)):
        sl = slice(off[i], off[i + 1])
        items.append((feats[sl], va[sl], int(labs[i])))
    return items


class ShardIterable(IterableDataset):
    def __init__(self, shards, shuffle=True, seed=0):
        self.shards = shards
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.shards)))
        if self.shuffle:
            rng.shuffle(order)
        for si in order:
            items = split_shard(self.shards[si])
            idx = list(range(len(items)))
            if self.shuffle:
                rng.shuffle(idx)
            for i in idx:
                f, v, lab = items[i]
                yield (torch.from_numpy(f.astype(np.float32)),
                       torch.from_numpy(v.astype(np.float32)), lab)


def collate(batch):
    feats, vas, labs = zip(*batch)
    T = max(f.shape[0] for f in feats)
    B, D = len(feats), feats[0].shape[1]
    nb = vas[0].shape[1]
    x = torch.zeros(B, T, D)
    va = torch.zeros(B, T, nb)
    mask = torch.zeros(B, T)
    for i, (f, v) in enumerate(zip(feats, vas)):
        t = f.shape[0]
        x[i, :t] = f
        va[i, :t] = v
        mask[i, :t] = 1.0
    return x, va, mask, torch.tensor(labs, dtype=torch.long)


def class_weights(shards):
    cnt = np.zeros(3)
    for s in shards:
        z = np.load(s)
        for c in z["labels"]:
            cnt[int(c)] += 1
    w = cnt.sum() / (3.0 * np.maximum(cnt, 1))
    print(f"クラス分布={cnt.astype(int).tolist()}  重み={w.round(3).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


def time_weight(mask, floor=0.1):
    """各発話で 0..1 のランプ（最終フレーム=1）。序盤を floor まで減衰。"""
    T = mask.shape[1]
    pos = torch.arange(1, T + 1, device=mask.device).float().unsqueeze(0)
    lengths = mask.sum(1, keepdim=True).clamp(min=1)
    w = (pos / lengths).clamp(max=1.0)
    w = floor + (1 - floor) * w
    return w * mask


@torch.no_grad()
def evaluate_lastframe(model, loader, device):
    model.eval()
    correct = total = 0
    for x, va, mask, lab in loader:
        x, mask, lab = x.to(device), mask.to(device), lab.to(device)
        cls, _ = model(x)
        last = mask.sum(1).long() - 1
        pred = cls[torch.arange(x.size(0)), last].argmax(-1)
        correct += (pred == lab).sum().item()
        total += x.size(0)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--dev_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_va", type=float, default=1.0)
    ap.add_argument("--d_hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tr = read_manifest(args.train_dir)
    dv = read_manifest(args.dev_dir)
    nb = np.load(tr[0])["va"].shape[1]
    print(f"train shards={len(tr)} dev shards={len(dv)} va_bins={nb}")
    cw = class_weights(tr).to(args.device)

    model = TurnTakingHead(d_hidden=args.d_hidden, n_layers=args.n_layers,
                           va_bins=nb).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ds = ShardIterable(tr, shuffle=True)
    dev_ds = ShardIterable(dv, shuffle=False)

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        tl_cls = tl_va = nstep = 0
        loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate)
        for x, va, mask, lab in loader:
            x, va, mask, lab = (x.to(args.device), va.to(args.device),
                                mask.to(args.device), lab.to(args.device))
            cls, vap = model(x)                                # (B,T,3),(B,T,nb)
            tw = time_weight(mask)                              # (B,T)
            tgt = lab.unsqueeze(1).expand(-1, x.size(1))        # (B,T)
            ce = F.cross_entropy(cls.transpose(1, 2), tgt, weight=cw,
                                 reduction="none")              # (B,T)
            cls_loss = (ce * tw).sum() / tw.sum().clamp(min=1)
            bce = F.binary_cross_entropy_with_logits(vap, va, reduction="none")
            va_loss = (bce * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1) / nb
            loss = cls_loss + args.lambda_va * va_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl_cls += cls_loss.item()
            tl_va += va_loss.item()
            nstep += 1
        dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, collate_fn=collate)
        acc = evaluate_lastframe(model, dev_loader, args.device)
        print(f"epoch {ep + 1:2d}  cls={tl_cls / nstep:.4f}  va={tl_va / nstep:.4f}  "
              f"dev_lastframe_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "va_bins": nb,
                        "d_hidden": args.d_hidden, "n_layers": args.n_layers},
                       args.out)
            print(f"  saved best -> {args.out}")
    print(f"完了. best dev_lastframe_acc={best:.4f}")


if __name__ == "__main__":
    main()
