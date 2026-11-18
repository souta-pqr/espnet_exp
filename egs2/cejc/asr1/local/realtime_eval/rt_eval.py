#!/usr/bin/env python
"""CEJC の録音（話者ごとのチャネル）を，デモと同じ実時間のプログラム（epd.engine）に頭から流し，応答した時刻を書く（2026-11-02）．
人が読み上げなくても，実時間の条件（silero-vad で決めた発話の頭・頭から約 1.5 秒はスコアが出ない・実際のスコアの遅れ・過去の文脈）で評価できる．
採点は rt_score.py（CEJC のラベルで，Full-Duplex-Bench の 3 課題と cut-in rate・latency）．

  cd ~/2026/work/exp-realtime-epd
  python <この場所>/rt_eval.py --wav_scp <data>/wav.scp --out <出力> --shard 0 --nshard 6 --model proposed,smart_turn_ft,silero
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())
from epd.vad import HOP, SR
from run_live import add_model_args, build

MAX_SEC = 30.0          # 提案手法が見る発話の長さの上限（直近の分だけ残す）．録音には他の人の声も入るので，silero が声ありを
                        # 続けて 1 発話が 160 秒にもなり，毎回の計算とメモリが長さの 2 乗で増えて機械を止めた（2026-11-03）
RESET_SEC = 120.0       # 提案手法のエンコーダの状態をこの長さごとに空にする（ふつうの発話には効かない）
MAX_RSS = 3.5e9         # 1 本のメモリの上限（超えたら自分で止まる．同じ機械の学習を守る）


def cap(scorer):
    """提案手法の push のあとで，エンコーダの出力を直近 MAX_SEC 秒に切る（スコアの時刻は統計にしか使わないのでずれてよい）"""
    if not hasattr(scorer, "frames"):
        return
    orig, m = scorer.push, int(MAX_SEC / scorer.sec_per_frame)

    def push(x, silent=True):
        if scorer.n_frames == 0:
            scorer.n_since = 0
        out = orig(x, silent)
        scorer.n_since = getattr(scorer, "n_since", 0) + len(out)
        if scorer.n_since * scorer.sec_per_frame > RESET_SEC:   # エンコーダの状態を空にする（何分も続く発話で位置の番号が
            scorer.est, scorer.n_since = None, 0                # 上限を超えて落ちた録音があった．11/03．T020_013b_IC01）
        if scorer.frames and scorer.frames[0].shape[1] > m:
            scorer.frames = [scorer.frames[0][:, -m:]]
            scorer.n_frames = m
        return out
    scorer.push = push


def restart(e, sil, carry=False):
    """発話の中で無音が sil 秒以上続いたあと声が戻ったら，そこで発話を区切り，新しい発話にする（2026-11-04）．
    実時間では，モデルが待つと発話が切れずに「前の区間＋間＋今の区間」がひとつながりで入るが，学習は区間の頭から始まる音声だけなので，
    それに合わせる．区切った前の発話は過去の文脈になる．"""
    v = e.vad
    e.prev_run = 0.0

    def vad(x):
        p, run = v(x)
        if e.in_utt and p >= 0.5 and e.prev_run >= sil:
            tail = e.utt[-e.pre.maxlen:]                  # 新しい発話の pre_roll は，前の発話の最後の音声
            e._end()
            e.pre.clear()
            e.pre.extend(tail)
        e.prev_run = run
        return p, run
    e.vad = vad
    if not carry:
        return
    feed = e.feed

    def feed2(x):
        """区切り直した発話のスコアがまだ出ないうちは，前の発話の最後のスコアを使う（他の人の声で区切り直したときに，
        応答が 3 秒の打ち切りまで遅れるのを防ぐ）"""
        s, b, was = getattr(e, "score", None), getattr(e, "p_bc", None), e.in_utt
        ev = feed(x)
        if was and e.in_utt and e.score is None:
            e.score, e.p_bc = s, b
        return ev
    e.feed = feed2


def rss():
    return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def run_one(a, rec, path):
    dst = f"{a.out}/{rec}.tsv"
    t0 = time.perf_counter()
    x, sr = sf.read(path, dtype="float32")
    assert sr == SR and x.ndim == 1, (path, sr, x.shape)
    systems = build(a)
    for _, _, e, _ in systems:
        cap(e.scorer)
        if a.sync_past and hasattr(e.scorer, "async_past"):
            e.scorer.async_past = False   # 過去の符号化を裏のスレッドでしない（同じエンコーダを逐次処理と同時に使って落ちた．11/04）
        if a.restart_sil is not None:
            restart(e, a.restart_sil, a.carry_score)
    ev = []
    for i in range(0, len(x) - HOP + 1, HOP):
        c = x[i:i + HOP]
        if i % (HOP * 1000) == 0 and rss() > MAX_RSS:
            sys.exit(f"{rec}: メモリが {rss() / 1e9:.1f} GB を超えたので止める")
        for n, _, e, _ in systems:
            r = e.feed(c)
            if r:
                ev.append((n, r))
    with open(dst + ".tmp", "w") as f:
        f.write("model\tt\tutt_start\tspeech_end\tcond\trun\tscore\tp_bc\n")
        for n, e in ev:
            f.write("\t".join([n] + [str(e[k]) for k in ("t", "utt_start", "speech_end", "cond", "run", "score", "p_bc")]) + "\n")
    os.replace(dst + ".tmp", dst)
    el = time.perf_counter() - t0
    print(f"{rec} {len(x) / SR / 60:.1f} 分を {el / 60:.1f} 分（実時間比 {el / (len(x) / SR):.2f}・メモリ {rss() / 1e9:.2f} GB）", flush=True)


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--wav_scp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--sync_past", action="store_true", help="提案手法：過去の文脈を発話の終わりにその場で符号化する")
    ap.add_argument("--carry_score", action="store_true", help="区切り直した発話のスコアが出るまで前のスコアを使う")
    ap.add_argument("--restart_sil", type=float, default=None, help="無音がこの秒数続いたあと声が戻ったら新しい発話にする")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [l.split() for l in open(a.wav_scp) if l.strip()]
    recs = [r for i, r in enumerate(sorted(recs)) if i % a.nshard == a.shard]
    ctx = mp.get_context("spawn")
    for rec, path in recs:
        if os.path.exists(f"{a.out}/{rec}.tsv"):
            continue
        p = ctx.Process(target=run_one, args=(a, rec, path))  # 録音ごとに別のプロセスで流す（同じプロセスで続けると，モデルを
        p.start()                                             # 作り直してもメモリが少しずつ残り，3.5 GB を超えて止まった．11/03）
        p.join()
        if p.exitcode:
            print(f"{rec}: 失敗（終了コード {p.exitcode}）", flush=True)


if __name__ == "__main__":
    main()
