#!/usr/bin/env python
"""RL halting policy（学習型停止方策）で早期確定を学習・評価する。

位置づけ（turntaking_related_problems.md 系統②／手法D・EARLIEST 系）:
  TEASER/SPRT は「学習済み cls の確率ストリーム p_t に後付けする固定規則」だった（系統①）。
  本スクリプトは「いつ止めるか」自体を**強化学習で獲得**する：
    - Discriminator = 学習済み発話区間末ヘッド（cls）。p_t を出す（ここは凍結・流用）。
    - Controller    = 各フレームで halt_prob_t を出す小さな MLP（これを REINFORCE で学習）。
    - 報酬 R = r_correct(ŷ_τ, y) − λ·(τ/(T−1))   （正解+1 / 誤り −c、遅延ペナルティ λ）

フルモデル（ASR+cls）は触らず、その出力（毎フレーム p_t）の上に停止方策を載せるので
学習が軽く、TEASER/SPRT と同じ p_t 上で「精度 × 確定位置%」を直接比較できる。

入力データは dump/raw/<set>_trail（utt 単位・trailing 込み、共有ストレージ不要）。
1) cls 確率ストリーム p_t を抽出してキャッシュ → 2) Controller を REINFORCE 学習 → 3) 評価。
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from evaluate_multitask import cls_stream, read_map  # noqa: E402


# ------------------------------------------------------------------
# 1. cls 確率ストリームの抽出（学習済みモデルを 1 回通してキャッシュ）
# ------------------------------------------------------------------
def extract_streams(config, model_path, data_dir, device, max_utts=0):
    from espnet2.tasks.asr import TurnTakingASRTask
    model, _ = TurnTakingASRTask.build_model_from_file(config, model_path, device)
    model.eval()

    data_dir = Path(data_dir)
    wav_scp = read_map(data_dir / "wav.scp")
    tags = {k: int(v) for k, v in read_map(data_dir / "tag").items()}
    items = list(wav_scp.items())
    if max_utts > 0:
        items = items[:max_utts]

    streams = []
    n = 0
    for utt, path in items:
        if "|" in path or utt not in tags:
            continue
        try:
            wav, _ = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            continue
        if wav.ndim > 1:
            wav = wav[:, 0]
        if len(wav) < 320:
            continue
        p = cls_stream(model, wav, device).astype(np.float16)   # (T,3) 省メモリ
        streams.append((p, tags[utt]))
        n += 1
        if n % 2000 == 0:
            print(f"  ...{n} 抽出", flush=True)
    print(f"抽出 {n} 区間: {data_dir}")
    return streams


def load_or_extract(cache, config, model_path, data_dir, device, max_utts):
    cache = Path(cache)
    if cache.is_file():
        print(f"キャッシュ読込: {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)
    s = extract_streams(config, model_path, data_dir, device, max_utts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(s, f)
    return s


# ------------------------------------------------------------------
# 2. Controller（停止方策）と特徴量
# ------------------------------------------------------------------
def featurize(p):
    """p: (T,3) 確率 → controller 入力 (T,6) = [p0,p1,p2, margin, entropy, pos]。"""
    p = p.astype(np.float32)
    T = p.shape[0]
    srt = np.sort(p, axis=1)
    margin = (srt[:, -1] - srt[:, -2])[:, None]
    ent = (-(p * np.log(p + 1e-8)).sum(1))[:, None]
    pos = (np.arange(T, dtype=np.float32) / max(T - 1, 1))[:, None]
    return np.concatenate([p, margin, ent, pos], axis=1)   # (T,6)


class Controller(nn.Module):
    def __init__(self, d_in=6, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):              # x: (T,6) -> halt_logit (T,)
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# 3. REINFORCE 学習
# ------------------------------------------------------------------
def sample_stop(halt_prob, train=True):
    """各フレームで停止をサンプル。最初に halt した frame τ を返す（無ければ最終）。"""
    T = halt_prob.shape[0]
    if train:
        u = torch.rand(T, device=halt_prob.device)
        fired = u < halt_prob
    else:
        fired = halt_prob > 0.5
    idx = torch.nonzero(fired, as_tuple=False)
    tau = int(idx[0]) if idx.numel() > 0 else T - 1
    return tau


def traj_logprob(halt_logit, tau):
    """軌跡（t<τ は continue, τ で halt）の対数確率。無発火（強制）なら全 continue。"""
    logp_halt = torch.nn.functional.logsigmoid(halt_logit)             # log σ
    logp_cont = torch.nn.functional.logsigmoid(-halt_logit)            # log(1-σ)
    T = halt_logit.shape[0]
    lp = logp_cont[:tau].sum()
    if tau < T - 1:                       # τ で実際に halt した
        lp = lp + logp_halt[tau]
    return lp


def train_reinforce(ctrl, streams, lam, cost, epochs, lr, batch, device):
    feats = [torch.from_numpy(featurize(p)).to(device) for p, _ in streams]
    labels = [y for _, y in streams]
    preds_arg = [np.asarray(p.astype(np.float32).argmax(1)) for p, _ in streams]
    opt = torch.optim.Adam(ctrl.parameters(), lr=lr)
    base = 0.0  # 報酬の移動平均ベースライン

    for ep in range(epochs):
        order = np.random.permutation(len(streams))
        tot_R, tot_pos, tot_acc, nb = 0.0, 0.0, 0.0, 0
        opt.zero_grad()
        loss_accum = 0.0
        for bi, i in enumerate(order):
            x = feats[i]
            T = x.shape[0]
            halt_logit = ctrl(x)
            halt_prob = torch.sigmoid(halt_logit)
            tau = sample_stop(halt_prob.detach(), train=True)
            yhat = preds_arg[i][tau]
            r_correct = 1.0 if yhat == labels[i] else -cost
            R = r_correct - lam * (tau / max(T - 1, 1))
            adv = R - base
            base = 0.99 * base + 0.01 * R
            lp = traj_logprob(halt_logit, tau)
            loss_accum = loss_accum + (-adv * lp)
            tot_R += R; tot_pos += tau / max(T - 1, 1); tot_acc += (yhat == labels[i]); nb += 1
            if (bi + 1) % batch == 0:
                (loss_accum / batch).backward()
                opt.step(); opt.zero_grad(); loss_accum = 0.0
        if isinstance(loss_accum, torch.Tensor):
            (loss_accum / batch).backward(); opt.step(); opt.zero_grad()
        print(f"  epoch {ep+1}/{epochs}: meanR={tot_R/nb:.3f} acc={tot_acc/nb:.3f} "
              f"確定位置={tot_pos/nb*100:.1f}%")


@torch.no_grad()
def evaluate(ctrl, streams, device):
    rec = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    correct = 0; pos = 0.0; n = 0
    for p, y in streams:
        x = torch.from_numpy(featurize(p)).to(device)
        halt_prob = torch.sigmoid(ctrl(x))
        tau = sample_stop(halt_prob, train=False)
        yhat = int(np.asarray(p.astype(np.float32))[tau].argmax())
        ok = int(yhat == y)
        correct += ok; rec[y][0] += ok; rec[y][1] += 1
        pos += (tau + 1) / p.shape[0]; n += 1
    acc = correct / max(n, 1)
    r = {c: rec[c][0] / max(rec[c][1], 1) for c in (0, 1, 2)}
    return acc, r, pos / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_dir", default="dump/raw/train_dev_trail")
    ap.add_argument("--eval_dir", default="dump/raw/eval_trail")
    ap.add_argument("--cache_dir", default="local/turntaking/rl_cache")
    ap.add_argument("--lambda_delay", type=float, default=1.0, help="遅延ペナルティ λ")
    ap.add_argument("--cost", type=float, default=1.0, help="誤判定ペナルティ c")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_utts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    tag = Path(args.model).stem
    cd = Path(args.cache_dir)
    tr = load_or_extract(cd / f"{tag}_train.pkl", args.config, args.model, args.train_dir, args.device, args.max_utts)
    ev = load_or_extract(cd / f"{tag}_eval.pkl", args.config, args.model, args.eval_dir, args.device, args.max_utts)

    ctrl = Controller().to(args.device)
    print(f"=== REINFORCE 学習 (λ={args.lambda_delay}, c={args.cost}) ===")
    train_reinforce(ctrl, tr, args.lambda_delay, args.cost, args.epochs, args.lr, args.batch, args.device)

    acc, r, pos = evaluate(ctrl, ev, args.device)
    print("=" * 60)
    print(f"【RL halting 評価 (eval)】λ={args.lambda_delay}")
    print(f"  acc={acc:.3f}  rec_no={r[0]:.3f} rec_yes={r[1]:.3f} rec_oth={r[2]:.3f}"
          f"  確定位置%={pos*100:.1f}")
    print("  ※ TEASER/SPRT（local/turntaking/evaluate_multitask.py）と同じ p_t 上の比較。")


if __name__ == "__main__":
    main()
