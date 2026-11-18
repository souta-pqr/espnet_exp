#!/usr/bin/env python
"""「です・ます」で終わる区間のラベルが，意味（伝えたいことを言い終えたか）で見て正しいかを確かめる（2026-10-16）。

人手正解 922 件と同じ材料（直前 10・後続 10 発話，重複タグ・ポーズ行・同話者継続間隔）を，
人手正解に入っていない LLM 判定の区間から作る。文末が「です・ます」系のものだけを無作為に抜く。

  python local/turntaking/desumasu_check.py --n 150 --out <dir>
出力：<dir>/items.jsonl（line_no・各ラベル・材料），<dir>/batch_XX.txt（読む用）
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vote_tags import read_votes                                     # noqa: E402

LAB = "/home/kobori/work/cejc-label"
spec = importlib.util.spec_from_file_location("det27", f"{LAB}/detect_section.text.py")
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)
RES = "/autofs/diamond5/share/users/kobori/result_oneshot.txt"
USR = "/autofs/diamond5/share/users/kobori/full_context/oneshot/prompt_user.txt"
R = "/autofs/diamond5/share/users/kobori/relabel/full_e/"
LR = re.compile(r"^\[LINE (\d+)\]")
UR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")
# 文末の形の分類（上から順に当てる）．「です・ます」系は後ろに「よ」が付くものも含む
CATS = [
    ("filler", None),                                                   # 末尾がフィラー・言い直し
    ("desumasu", re.compile(r"(です|ます|でした|ました|ません|ませんでした)(よ)?$")),
    ("question", re.compile(r"(か|の|かな|っけ|って)$")),
    ("agree", re.compile(r"(ね|よね|じゃん|でしょ|でしょう|じゃない|もんね|な)$")),
    ("connect", re.compile(r"(て|で|けど|けれど|けども|から|ので|し|が|のに|たら|ば|って)$")),
    ("plain", re.compile(r"(た|だ|よ|る|う|く|す|つ|む|ぶ|ぐ|い|わ|ぞ|ぜ|さ)$")),
]


def category(text):
    raw = re.sub(r"</?[^>]*と重複>", "", text).strip()
    if re.search(r"\((?:F|D)[^)]*\)\s*$", raw):
        return "filler"
    c = clean(text)
    for nm, rx in CATS[1:]:
        if rx.search(c):
            return nm
    return "other"


def clean(t):
    t = re.sub(r"</?[^>]*と重複>", "", t)
    t = re.sub(r"\((?:F|D)[^)]*\)\s*$", "", t.strip())       # 末尾のフィラー・言い直しは文末の形の判定から外す
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cats", default="desumasu", help="抜き出す文末の分類（空白区切り）")
    a = ap.parse_args()
    gold = {int(l.split("\t")[1]) for l in list(open("eval-1000/gold.txt"))[1:]}
    d = det.LLMDetector.__new__(det.LLMDetector)
    d.prompt_template = open(USR, encoding="utf-8").read()
    cand = []; cur = None; blk = None; sec = None; n_llm = 0
    for raw in open(RES, encoding="utf-8"):
        if raw.startswith("[LINE"):
            cur = int(LR.match(raw)[1]); blk = {"h": [], "t": None, "p": []}; sec = None; continue
        if cur is None:
            continue
        if raw.startswith("[HISTORY]"): sec = "h"; continue
        if raw.startswith("[TARGET]"): sec = "t"; continue
        if raw.startswith("[POST_HISTORY]"): sec = "p"; continue
        if raw.startswith("[RESULT]"):
            t = blk["t"]
            if t is not None and not det.is_aizuchi(t) and not det.is_turn_end_next_different_speaker(t, blk["p"]):
                n_llm += 1
                if cur not in gold:
                    cand.append((cur, int(raw.split()[1]), blk, category(t.text)))
            cur = None; continue
        m = UR.match(raw.rstrip("\n"))
        if m and sec:
            u = det.EvalUtterance(*m.groups())
            if sec == "t": blk["t"] = u
            else: blk[sec].append(u)
    print(f"LLM 判定の区間 {n_llm}，うち人手正解以外 {len(cand)}")
    v32 = read_votes(R + "Qwen3-32B-AWQ-nothink.add_e.tsv")
    v8 = read_votes(R + "Qwen3-8B-think.add_e.tsv")
    vg = read_votes(R + "gemma-4-31B-it-think.add_e.tsv")
    cand = [c for c in cand if c[0] in v32 and c[0] in v8 and c[0] in vg]
    # 文末の形ごとの，母集団でのラベルの割合
    import collections
    by = collections.defaultdict(list)
    for c in cand:
        by[c[3]].append(c)
    print("文末の形      区間数   完了とした割合（gemma / 多数決）")
    for nm in [c[0] for c in CATS] + ["other"]:
        L = by[nm]
        if L:
            g = sum(vg[c[0]] for c in L) / len(L); v = sum(int(c[1] + v32[c[0]] + v8[c[0]] >= 2) for c in L) / len(L)
            print(f"  {nm:10s} {len(L):7d}   {g:.3f} / {v:.3f}")
    random.seed(a.seed)
    samp = []
    for nm in a.cats.split():
        samp += random.sample(by[nm], a.n)
    samp = sorted(samp, key=lambda c: (c[3], c[0]))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for i, (ln, q, blk, cat) in enumerate(samp):
            u = d.make_prompt(blk["h"], blk["t"], blk["p"])
            u = u[u.rindex("---") + 3:]                       # one-shot の例を除き，判定する部分だけ
            u = u[:u.rindex("以下の1行")].strip()
            f.write(json.dumps({"id": i, "line_no": ln, "cat": cat, "qwen": q, "q32": v32[ln], "q8t": v8[ln],
                                "gemma": vg[ln], "vote": int(q + v32[ln] + v8[ln] >= 2),
                                "material": u}, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in open(out / "items.jsonl", encoding="utf-8")]
    for b in range(0, len(rows), 25):     # 読む用（ラベルは書かない）
        with open(out / f"batch_{b // 25:02d}.txt", "w", encoding="utf-8") as f:
            for r in rows[b:b + 25]:
                f.write(f"=== 項目 {r['id']} ===\n{r['material']}\n\n")
    print(f"{a.n} 件 → {out}")


if __name__ == "__main__":
    main()
