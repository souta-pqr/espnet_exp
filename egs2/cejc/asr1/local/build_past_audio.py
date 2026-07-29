#!/usr/bin/env python
"""seqcat 方式用: 各発話の「直前N発話（同一話者）の音声を連結した past_speech.scp」を作る。

past_speech は sox パイプで過去N発話の音声を連結（古→新の時系列順）。ストレージ増なし。
モデル側で past_speech の後ろに現発話を連結し、エンコーダに通して現発話フレームを pool する。
過去が無い発話（会話先頭など）は 20ms の無音を割り当てる（＝実質 現発話のみ）。

同一話者では「現発話の start より前」＝「end より前」（過去は必ず現発話より前に終わる）。
出力: <dump_dir>/<out_name>.scp
      <dump_dir>/<out_name>_bounds.scp  … 連結した各過去発話のサンプル数（発話境界）

境界は「連結波形のどこで発話が切り替わるか」で、モデル側で発話単位の圧縮
(tt_past_pool="utt") に使う。連結してしまうと切れ目が失われるため別途出力する。
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path


def read_map(path):
    m = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(maxsplit=1)
            if len(p) == 2:
                m[p[0]] = p[1].strip()
    return m


def session_of(reco):
    return re.sub(r"_IC\d+$", "", reco)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True, help="data/<set>（segments）")
    ap.add_argument("--dump_dir", required=True, help="dump/raw/<set>（wav.scp・出力先）")
    ap.add_argument("--scope", choices=["same", "session"], default="same")
    ap.add_argument("--n_past", type=int, default=2)
    ap.add_argument("--max_past_sec", type=float, default=30.0)
    ap.add_argument("--out_name", default="past_speech")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--silence", default="data/silence_20ms.wav",
                    help="過去なし発話に割り当てる無音ファイル")
    args = ap.parse_args()

    wav = read_map(Path(args.dump_dir) / "wav.scp")
    nsamp = {k: int(v) for k, v in
             read_map(Path(args.dump_dir) / "utt2num_samples").items()}
    seg = {}
    with open(Path(args.seg_dir) / "segments", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                seg[p[0]] = (p[1], float(p[2]), float(p[3]))   # reco, start, end

    utts = [u for u in wav if u in seg]
    groups = defaultdict(list)
    for u in utts:
        reco = seg[u][0]
        key = reco if args.scope == "same" else session_of(reco)
        groups[key].append(u)
    for k in groups:
        groups[k].sort(key=lambda x: seg[x][1])       # start 昇順
    idx = {u: i for k in groups for i, u in enumerate(groups[k])}

    out = Path(args.dump_dir) / f"{args.out_name}.scp"
    out_b = Path(args.dump_dir) / f"{args.out_name}_bounds.scp"
    n_nopast = 0
    with open(out, "w", encoding="utf-8") as f, open(out_b, "w", encoding="utf-8") as fb:
        for u in utts:
            reco, s, e = seg[u]
            key = reco if args.scope == "same" else session_of(reco)
            sib = groups[key]
            i = idx[u]
            past = []
            # 直前N発話（start 昇順で i の手前）を max_past 窓内で集める
            for j in range(i - 1, max(-1, i - 1 - args.n_past), -1):
                pu = sib[j]
                if seg[pu][1] >= s - args.max_past_sec:
                    past.append(pu)
                else:
                    break
            past.reverse()                             # 古→新（時系列順）
            if past:
                # 絶対パスの列挙のみ（sox 等の subprocess は使わない＝fork worker で安全）
                # 読み込み側 (concat_sound ローダ) が soundfile で読んで時間連結する
                files = " ".join(str(Path(wav[p]).resolve()) for p in past)
                # 連結順に各過去発話のサンプル数を並べる（＝発話境界）
                bounds = " ".join(str(nsamp[p]) for p in past)
            else:
                files = str(Path(args.silence).resolve())   # 過去なし＝20ms無音
                bounds = str(int(0.02 * args.sample_rate))
                n_nopast += 1
            f.write(f"{u} {files}\n")
            fb.write(f"{u} {bounds}\n")
    print(f"{out}: {len(utts)}発話 (過去なし={n_nopast}, scope={args.scope}, "
          f"N={args.n_past}, max_past={args.max_past_sec}s)")
    print(f"{out_b}: 発話境界（連結順の各過去発話サンプル数）")


if __name__ == "__main__":
    main()
