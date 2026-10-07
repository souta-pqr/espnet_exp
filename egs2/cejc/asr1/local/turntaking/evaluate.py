#!/usr/bin/env python
"""学習済み turn-taking ヘッドを停止規則つきで評価.

毎フレームの事後確率ストリームに対し、早期確定規則を適用:
  TEASER : 最大確率 p_max>θ が v フレーム連続したら、その時点のクラスで確定。
  SPRT   : 対数尤度比の累積（簡易版: log p(top) - log p(2nd) を加算）が境界Aを超えたら確定。
発火しなければ最終フレームで確定（=全部待つ最遅ケース）。
出力: 精度(acc/クラス別recall) と 早期性(発話のどこで確定したか平均%、待ったフレーム数).
θ を sweep して「精度 vs 遅延」曲線を出す。
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from model import TurnTakingHead
from train import read_manifest, split_shard


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location=device)
    m = TurnTakingHead(d_hidden=st["d_hidden"], n_layers=st["n_layers"],
                       va_bins=st["va_bins"]).to(device)
    m.load_state_dict(st["model"])
    m.eval()
    return m


@torch.no_grad()
def posteriors(model, shards, device):
    """各発話の (prob[T,3], label) を列挙."""
    for s in shards:
        for f, _v, lab in split_shard(s):
            x = torch.from_numpy(f.astype(np.float32)).unsqueeze(0).to(device)
            cls, _ = model(x)
            p = torch.softmax(cls[0], -1).cpu().numpy()        # (T,3)
            yield p, lab


def fire_teaser(p, theta, v):
    """p_max>theta が v 連続で発火。(pred, fire_idx). 無発火なら最終フレーム。"""
    pred = p.argmax(-1)
    pmax = p.max(-1)
    run = 0
    for t in range(len(p)):
        if pmax[t] > theta and (run == 0 or pred[t] == pred[t - 1]):
            run += 1
        else:
            run = 1 if pmax[t] > theta else 0
        if run >= v:
            return int(pred[t]), t
    return int(pred[-1]), len(p) - 1


def fire_sprt(p, bound, eps=1e-6):
    """簡易SPRT: 各フレームで log(top)-log(2nd) を累積、bound 超で確定."""
    lp = np.log(p + eps)
    acc = np.zeros(3)
    for t in range(len(p)):
        acc += lp[t]
        order = acc.argsort()
        margin = acc[order[-1]] - acc[order[-2]]
        if margin > bound:
            return int(order[-1]), t
    return int(acc.argmax()), len(p) - 1


def run_rule(streams, rule, param, v=2):
    n = len(streams)
    correct = 0
    recall = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    fracs = []
    for p, lab in streams:
        if rule == "teaser":
            pred, idx = fire_teaser(p, param, v)
        else:
            pred, idx = fire_sprt(p, param)
        ok = int(pred == lab)
        correct += ok
        recall[lab][0] += ok
        recall[lab][1] += 1
        fracs.append((idx + 1) / len(p))
    acc = correct / max(n, 1)
    rec = {c: recall[c][0] / max(recall[c][1], 1) for c in (0, 1, 2)}
    return acc, rec, float(np.mean(fracs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--rule", choices=["teaser", "sprt"], default="teaser")
    ap.add_argument("--v", type=int, default=2, help="TEASER 連続フレーム数")
    ap.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--sprt_bounds", default="1.0,2.0,4.0,8.0,16.0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    shards = read_manifest(args.eval_dir)
    print("事後確率ストリーム生成中...")
    streams = list(posteriors(model, shards, args.device))
    print(f"発話数={len(streams)}")

    params = ([float(x) for x in args.thresholds.split(",")] if args.rule == "teaser"
              else [float(x) for x in args.sprt_bounds.split(",")])
    pname = "θ" if args.rule == "teaser" else "bound"
    print("=" * 72)
    print(f"停止規則={args.rule}  (v={args.v})   精度 vs 早期性")
    print(f"{pname:>7} | {'acc':>6} | {'rec_no':>7} {'rec_yes':>7} {'rec_oth':>7} | "
          f"{'確定位置%':>8}")
    print("-" * 72)
    for pr in params:
        acc, rec, frac = run_rule(streams, args.rule, pr, args.v)
        print(f"{pr:7.2f} | {acc:6.3f} | {rec[0]:7.3f} {rec[1]:7.3f} {rec[2]:7.3f} | "
              f"{frac * 100:7.1f}%")
    print("=" * 72)
    print("確定位置%が小さいほど早期確定（遅延小）。θ/boundを上げると精度↑遅延↑。")


if __name__ == "__main__":
    main()
