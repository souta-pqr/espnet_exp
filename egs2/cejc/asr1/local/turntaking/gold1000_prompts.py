#!/usr/bin/env python
"""人手正解 1000 件について、Qwen3（exp27 full_context/oneshot）と同じ入力のプロンプトを作り直す。

ラベル付けの手順は ~/work/cejc-label/detect_section.text.py と同じ:
  1. 中身が相槌の一覧（AIZUCHI_SET）に入る       → 2（ルール。LLM を使わない）
  2. 直後に話し始めるのが別の話者                → 1（ルール。LLM を使わない）
  3. それ以外だけ LLM が「はい／いいえ」で判定    → 1 / 0
文脈（直前 10・後続 10 発話）は Qwen3 の結果ファイルに残っているものをそのまま使い、
描画（重複タグ・ポーズ行・同話者継続間隔）は exp27 のスクリプト、テンプレートは full_context/oneshot のものをそのまま使う。

  python local/turntaking/gold1000_prompts.py
"""
import importlib.util
import json
import re

LAB = "/home/kobori/work/cejc-label"
spec = importlib.util.spec_from_file_location("det27", f"{LAB}/detect_section.text.py")
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)

RES = "/autofs/diamond5/share/users/kobori/result_oneshot.txt"
SYS = "/autofs/diamond5/share/users/kobori/full_context/oneshot/prompt_system.txt"
USR = "/autofs/diamond5/share/users/kobori/full_context/oneshot/prompt_user.txt"
GOLD = "eval-1000/gold.txt"
OUT = "tt_analysis/gold1000_prompts.jsonl"
LR = re.compile(r"^\[LINE (\d+)\]")
UR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")


def main():
    gold = {int(l.split("\t")[1]): l.rstrip("\n").split("\t") for l in list(open(GOLD))[1:]}
    items = {}; cur = None; sec = None; blk = None
    for raw in open(RES, encoding="utf-8"):
        if raw.startswith("[LINE"):
            ln = int(LR.match(raw)[1]); cur = ln if ln in gold else None
            blk = {"h": [], "t": None, "p": []}; sec = None
            continue
        if cur is None:
            continue
        if raw.startswith("[HISTORY]"): sec = "h"; continue
        if raw.startswith("[TARGET]"): sec = "t"; continue
        if raw.startswith("[POST_HISTORY]"): sec = "p"; continue
        if raw.startswith("[RESULT]"):
            items[cur] = (blk, int(raw.split()[1])); cur = None; continue
        m = UR.match(raw.rstrip("\n"))
        if m and sec:
            u = det.EvalUtterance(*m.groups())
            if sec == "t": blk["t"] = u
            else: blk[sec].append(u)
    sys_p = open(SYS, encoding="utf-8").read().strip()
    d = det.LLMDetector.__new__(det.LLMDetector)        # モデルは読まずにプロンプト作成だけ使う
    d.prompt_template = open(USR, encoding="utf-8").read()
    with open(OUT, "w", encoding="utf-8") as f:
        for ln in sorted(items):
            blk, qwen = items[ln]
            h, t, p = blk["h"], blk["t"], blk["p"]
            if det.is_aizuchi(t):
                route = "rule_aizuchi"
            elif det.is_turn_end_next_different_speaker(t, p):
                route = "rule_nextdiff"
            else:
                route = "llm"
            g = gold[ln]
            tgt = f"[{det.speaker_short(t.speaker)} | {float(t.end) - float(t.start):.1f}秒] {t.text}"
            f.write(json.dumps({"line_no": ln, "rank": int(g[0]), "gold": int(g[3]), "qwen": qwen,
                                "route": route, "target_ok": tgt == g[4], "system": sys_p,
                                "user": d.make_prompt(h, t, p)}, ensure_ascii=False) + "\n")
    print(f"{len(items)} 件 → {OUT}")


if __name__ == "__main__":
    main()
