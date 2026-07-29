#!/usr/bin/env python
"""過去発話を「テキスト」で入れる版の入力を作る。

各発話について、直前N発話（same=同一話者 / session=同一セッション）の書き起こしを
古→新の時系列順に連結し、token_list でトークンID列に変換して出力する。
出力: <dump_dir>/<out_name>.scp  各行「uttid id1 id2 ...」（text_int で読める）

過去なし（会話先頭など）の発話は空 → 学習側で長さ0（過去なし）扱い。
書き起こしにはフィラー(F ...)・言い直し(D ...) のタグがそのまま含まれる（意図的に残す）。
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
    ap.add_argument("--seg_dir", required=True, help="data/<set>（segments と text）")
    ap.add_argument("--dump_dir", required=True, help="出力先 dump/raw/<set>")
    ap.add_argument("--token_list", default="data/jp_token_list/word/tokens.txt")
    ap.add_argument("--scope", choices=["same", "session"], default="same")
    ap.add_argument("--n_past", type=int, default=2)
    ap.add_argument("--max_past_sec", type=float, default=30.0)
    ap.add_argument("--out_name", default="past_text")
    args = ap.parse_args()

    # token -> id
    tok2id = {}
    with open(args.token_list, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok2id[line.rstrip("\n")] = i
    unk = tok2id.get("<unk>", 1)

    seg = {}
    with open(Path(args.seg_dir) / "segments", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                seg[p[0]] = (p[1], float(p[2]), float(p[3]))   # reco, start, end
    text = read_map(Path(args.seg_dir) / "text")

    # 出力対象は dump の wav.scp に実在する発話に限定（dump で除外された発話を出さない＝キー一致）
    dump_wav = read_map(Path(args.dump_dir) / "wav.scp")
    utts = [u for u in seg if u in text and u in dump_wav]
    groups = defaultdict(list)
    for u in utts:
        reco = seg[u][0]
        key = reco if args.scope == "same" else session_of(reco)
        groups[key].append(u)
    for k in groups:
        groups[k].sort(key=lambda x: seg[x][1])
    idx = {u: i for k in groups for i, u in enumerate(groups[k])}

    out = Path(args.dump_dir) / f"{args.out_name}.scp"
    n_nopast = 0
    # scp はキー昇順でソート必須（ESPnet のデータローダが複数 scp のキー一致を要求するため）
    utts = sorted(utts)
    with open(out, "w", encoding="utf-8") as f:
        for u in utts:
            reco, s, e = seg[u]
            key = reco if args.scope == "same" else session_of(reco)
            sib = groups[key]
            i = idx[u]
            past = []
            for j in range(i - 1, max(-1, i - 1 - args.n_past), -1):
                pu = sib[j]
                if seg[pu][1] >= s - args.max_past_sec:
                    past.append(pu)
            past.reverse()                                  # 古→新
            toks = []
            for pu in past:
                toks.extend(text[pu].split())
            ids = [str(tok2id.get(t, unk)) for t in toks]
            if not ids:
                n_nopast += 1
                # 過去なし: 行を出さない（学習側で欠損＝長さ0扱い）… ではなく
                # text_int は全 utt 必要なので、空を避けるため <unk> 1 個で埋める手もあるが、
                # ここでは 0（=<blank>）1個を置き「ほぼ無情報」にする。
                ids = ["0"]
            f.write(f"{u} {' '.join(ids)}\n")
    print(f"出力: {out}  (過去なし={n_nopast}/{len(utts)}, N={args.n_past}, scope={args.scope})")


if __name__ == "__main__":
    main()
