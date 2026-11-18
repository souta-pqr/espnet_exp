#!/usr/bin/env python
"""result_oneshot.txt の全 [LINE n] から、LLM が判定する区間とそのプロンプトを作る（段階 2）。

gold1000_prompts.py と同じ手順:
  is_aizuchi(対象)                                  → rule_aizuchi（対象外）
  is_turn_end_next_different_speaker(対象, 後続)    → rule_nextdiff（対象外）
  それ以外                                          → llm（判定する）
プロンプトは full_context/oneshot のテンプレートを LLMDetector.make_prompt で描画する。

  python full_items.py --check   # 件数の集計と、gold1000_prompts.jsonl の 922 件との文字列一致の確認
"""
import argparse
import collections
import importlib.util
import json
import re

SHARE = "/autofs/diamond5/share/users/kobori"
DET = f"{SHARE}/full_context/detect_section_text.py"
RES = f"{SHARE}/result_oneshot.txt"
SYS = f"{SHARE}/full_context/oneshot/prompt_system.txt"
USR = f"{SHARE}/full_context/oneshot/prompt_user.txt"
GOLD = f"{SHARE}/relabel/gold1000_prompts.jsonl"
LR = re.compile(r"^\[LINE (\d+)\]")
UR = re.compile(r"^- 開始=(\S+)\s+終了=(\S+)\s+話者=(\S+)\s+発話=(.+?)\s*$")

spec = importlib.util.spec_from_file_location("det", DET)
det = importlib.util.module_from_spec(spec); spec.loader.exec_module(det)


def system_prompt():
    return open(SYS, encoding="utf-8").read().strip()


def detector():
    d = det.LLMDetector.__new__(det.LLMDetector)        # モデルは読まずにプロンプト作成だけ使う
    d.prompt_template = open(USR, encoding="utf-8").read()
    return d


def iter_blocks(path=RES):
    """(line_no, route, history, target, post_history) を [LINE] の順に返す。"""
    cur = None; sec = None; blk = None
    for raw in open(path, encoding="utf-8"):
        if raw.startswith("[LINE"):
            cur = int(LR.match(raw)[1]); blk = {"h": [], "t": None, "p": []}; sec = None
            continue
        if cur is None:
            continue
        if raw.startswith("[HISTORY]"): sec = "h"; continue
        if raw.startswith("[TARGET]"): sec = "t"; continue
        if raw.startswith("[POST_HISTORY]"): sec = "p"; continue
        if raw.startswith("[RESULT]"):
            h, t, p = blk["h"], blk["t"], blk["p"]
            if det.is_aizuchi(t):
                route = "rule_aizuchi"
            elif det.is_turn_end_next_different_speaker(t, p):
                route = "rule_nextdiff"
            else:
                route = "llm"
            yield cur, route, h, t, p
            cur = None
            continue
        m = UR.match(raw.rstrip("\n"))
        if m and sec:
            u = det.EvalUtterance(*m.groups())
            if sec == "t": blk["t"] = u
            else: blk[sec].append(u)


def iter_llm_prompts(path=RES, routes=("llm",)):
    """LLM が判定する区間だけ (line_no, user プロンプト) を返す。
    routes に "rule_nextdiff" を足すと，直後に別の話者が話し始める区間も判定する（2026-10-16．
    規則で「完了」にしていたが，人手正解 78 件中 17 件が継続だったため）。"""
    d = detector()
    for ln, route, h, t, p in iter_blocks(path):
        if route in routes:
            yield ln, d.make_prompt(h, t, p)


def check():
    gold = {}
    for l in open(GOLD, encoding="utf-8"):
        r = json.loads(l)
        gold[r["line_no"]] = r
    sys_p = system_prompt()
    d = detector()
    cnt = collections.Counter(); seen = 0; same = 0; diff = []; route_diff = []
    for ln, route, h, t, p in iter_blocks():
        cnt[route] += 1
        if ln in gold:
            g = gold[ln]
            if g["route"] != route:
                route_diff.append(ln)
            if route == "llm":
                seen += 1
                if g["system"] == sys_p and g["user"] == d.make_prompt(h, t, p):
                    same += 1
                else:
                    diff.append(ln)
    print(f"[LINE] 合計 {sum(cnt.values())} 件")
    for k in ("rule_aizuchi", "rule_nextdiff", "llm"):
        print(f"  {k:14s} {cnt[k]:7d}")
    print(f"gold 1000 件との経路の食い違い: {len(route_diff)} 件 {route_diff[:10]}")
    n_gold_llm = sum(r["route"] == "llm" for r in gold.values())
    print(f"gold の llm {n_gold_llm} 件のうち 見つかった {seen} / 文字列が完全一致 {same} / 不一致 {len(diff)} {diff[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        check()
