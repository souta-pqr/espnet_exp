#!/usr/bin/env python
"""人手正解を足すための判定用紙を作る（2026-10-16）。

今の人手正解 1000 件は，LLM が判定する区間が 922 件で，規則で決めている区間がほとんど入っていない
（規則で完了 78 件，相槌 0 件。全体では規則で完了 148,961，相槌 110,251，LLM 137,812）。
規則の正しさを測るため，規則で「完了」にした区間と「相槌」にした区間から無作為に抜く。
材料は人手正解と同じ（前後 10 発話，重複タグ・ポーズ行・同話者継続間隔）。どちらの規則かは伏せ，順番を混ぜる。

  python local/turntaking/gold_add_sheet.py --n 100 --out tt_analysis/gold_add_20261016
出力：<out>/sheet.csv（表計算ソフト用．判定の列に 0=継続 / 1=完了 / 2=相槌 を記入）
      <out>/sheet.txt（読む用）
      <out>/key.tsv（番号 → line_no・規則・今のラベル．判定が終わるまで見ない）
"""
import argparse
import csv
import importlib.util
import random
import re
from pathlib import Path

LAB = "/home/kobori/work/cejc-label"
spec = importlib.util.spec_from_file_location("det27", f"{LAB}/detect_section.text.py")
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)
RES = "/autofs/diamond5/share/users/kobori/result_oneshot.txt"
USR = "/autofs/diamond5/share/users/kobori/full_context/oneshot/prompt_user.txt"
LR = re.compile(r"^\[LINE (\d+)\]")
UR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")
HEAD = """判定の基準：判定対象発話の終わりの時点で，
  1 = 完了：話し手が伝えたいことを言い終えた（ここで相手が話し始めても遮ることにならない）
  0 = 継続：まだ続きを話すつもりで，途中で区切れているだけ
  2 = 相槌：相手の話を聞いていることを示す短い反応（うん・へー・そうそう 等）で，自分の番を取らないもの
文末の形だけで決めず，前後の流れから判断する。後続の発話履歴も参考にしてよい。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="規則ごとの件数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gold = {int(l.split("\t")[1]) for l in list(open("eval-1000/gold.txt"))[1:]}
    d = det.LLMDetector.__new__(det.LLMDetector)
    d.prompt_template = open(USR, encoding="utf-8").read()
    pool = {"rule_nextdiff": [], "rule_aizuchi": []}
    cur = None; blk = None; sec = None
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
            if t is not None and cur not in gold:
                if det.is_aizuchi(t):
                    pool["rule_aizuchi"].append((cur, blk))
                elif det.is_turn_end_next_different_speaker(t, blk["p"]):
                    pool["rule_nextdiff"].append((cur, blk))
            cur = None; continue
        m = UR.match(raw.rstrip("\n"))
        if m and sec:
            u = det.EvalUtterance(*m.groups())
            if sec == "t": blk["t"] = u
            else: blk[sec].append(u)
    rng = random.Random(a.seed)
    samp = [(r, ln, blk) for r in pool for ln, blk in rng.sample(pool[r], a.n)]
    rng.shuffle(samp)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cur_lab = {"rule_nextdiff": 1, "rule_aizuchi": 2}
    with open(out / "sheet.csv", "w", encoding="utf-8-sig", newline="") as fc, \
         open(out / "sheet.txt", "w", encoding="utf-8") as ft, \
         open(out / "key.tsv", "w", encoding="utf-8") as fk:
        w = csv.writer(fc)
        w.writerow(["番号", "判定対象発話", "材料（前後の発話）", "判定（0=継続 / 1=完了 / 2=相槌）", "メモ"])
        ft.write(HEAD + "\n")
        fk.write("番号\tline_no\t規則\t今のラベル\n")
        for i, (r, ln, blk) in enumerate(samp, 1):
            u = d.make_prompt(blk["h"], blk["t"], blk["p"])
            u = u[u.rindex("---") + 3:]                       # one-shot の例を除く
            u = u[:u.rindex("以下の1行")].strip()
            tgt = re.sub(r"</?[^>]*と重複>", "", blk["t"].text)
            w.writerow([i, tgt, u, "", ""])
            ft.write(f"=== 番号 {i} ===\n{u}\n\n判定（0=継続 / 1=完了 / 2=相槌）：\n\n")
            fk.write(f"{i}\t{ln}\t{r}\t{cur_lab[r]}\n")
    print(f"規則で完了 {len(pool['rule_nextdiff'])}・相槌 {len(pool['rule_aizuchi'])} から各 {a.n} 件 → {out}")


if __name__ == "__main__":
    main()
