#!/usr/bin/env python
"""ラベル付けのプロンプトが難しすぎないかを確かめるため、人手正解 922 件の簡単なプロンプトを作る（2026-10-09）。

元のプロンプト（tt_analysis/gold1000_prompts.jsonl）は判断基準が (a)〜(d) の規則の列挙で、
重なりを表すタグ <○○と重複> が入れ子になる。次の 3 通りを作る。

  def        … 判断基準を「伝えたいことを言い終えたか」という定義だけに置き換える（例の根拠も定義で書き直す）
  noovl      … 判断基準はそのままで、重なりのタグだけ外す
  def_noovl  … 両方
  add_e      … 判断基準はそのままで、(e) 普通の言い切り・倒置・体言止めを「はい」に加える
               （人手との食い違いの分析で、言い切りを「いいえ」にする誤りが最も多かったため）
  add_f      … add_e に (f) 言いさし・倒置・引用末・語句だけの問い返しを加え、文末の形でなく
               同じ話者が続きを話しているかで決めるよう注意書きを直す（2026-10-16，270 件の確認で
               gemma がこれらを「いいえ」にしがちだったため）

出力: <outdir>/gold1000_prompts.<variant>.jsonl（元と同じ形式。route=="llm" の行だけ）

  python local/turntaking/gold1000_prompt_variants.py --outdir /autofs/diamond5/share/users/kobori/relabel/prompt_check
"""
import argparse
import json
import re
from pathlib import Path

SYSTEM_DEF = """以下は日本語会話の一部。

【判定対象発話】の終わりの時点で、話し手が伝えたいことを言い終えたかを判定する。
- 言い終えた（ここで相手が話し始めても、話し手の話を遮ることにならない）→「はい」
- まだ続きを話すつもりで、途中で区切れているだけ →「いいえ」
文末の形だけで決めず、前後の会話の流れから話し手の意図を読み取って判断すること。後続の発話履歴も参考にしてよい。

記号：(F 〜) フィラー、(D 〜) 言い直し、(R 〜) 伏せた固有名詞、<unclear> 聞き取れない箇所、〔ポーズ: X.X秒〕 発話間の無音。
「同話者継続間隔」は、判定対象発話の後に同じ話者が再び話すまでの時間。

以下の1行だけを出力すること。余計な前置きやコードブロックは禁止。
判定: はい または いいえ"""

REASON_DEF = {
    "【判定の根拠】「逃走したんだよね」は条件(c)「〜だよね」に当たるため「はい」。":
        "【判定の根拠】「逃走したんだよね」で伝えたいことを言い終えており、相手が質問で応じている。言い終えているので「はい」。",
    "【判定の根拠】「〜ってゆう(F あの)」はフィラーで終わり発話が途中。(a)(b)(c)(d)に当たらないため「いいえ」。":
        "【判定の根拠】「〜ってゆう(F あの)」で話が途中で切れており、同じ話者がすぐに続きを話している。まだ言い終えていないので「いいえ」。",
}
HEAD_DEF = {
    "【例1】（「〜だよね」でターンを譲る形）": "【例1】（言い終えている）",
    "【例2】（フィラーで終わり、同話者が直後に継続）": "【例2】（まだ途中）",
}
E_OLD = "■判断基準：判定対象発話が次の(a)〜(d)のどれかに当てはまれば「はい」、どれにも当てはまらなければ「いいえ」。"
E_NEW = "■判断基準：判定対象発話が次の(a)〜(e)のどれかに当てはまれば「はい」、どれにも当てはまらなければ「いいえ」。"
# 例文は人手正解 922 件に出てこないものにする（答えを見せないため）
E_ADD = ("(e) 普通の言い切り・まとまった発話：「〜た」「〜だ」「〜です」「〜ます」「〜よ」などで言い切っている、"
         "倒置（「おいしかったよあの店」）や体言止め（「来週の水曜日」）で言いたいことを言い終えている、"
         "「了解です」「それは困るね」のような応答・感想として完結している場合")
E_NOTE_OLD = "※「いいえ」になるのは、(a)(b)(c)(d)のどれにも当てはまらない場合。"
E_NOTE_NEW = "※「いいえ」になるのは、(a)〜(e)のどれにも当てはまらない場合。"
F_NEW = "■判断基準：判定対象発話が次の(a)〜(f)のどれかに当てはまれば「はい」、どれにも当てはまらなければ「いいえ」。"
# 例文は人手正解 922 件にも，確認に使った 270 件にも出てこないものにする
F_ADD = ("(f) 言いさし・倒置・引用で終わっていても、言いたいことを言い終えている発話：「〜けど」「〜から」「〜し」で区切って"
         "後ろを言わずに済ませている（「そこまでは言わないけど」「あとは私がやるから」）、「〜ってゆう」「〜って」で感想や"
         "引用を言い終えている、「え、明日」のように語句だけで問い返している場合。"
         "ただし、後続の発話履歴で同じ話者がその文の続きを話していれば当てはまらない")
F_NOTE_OLD = ("※「いいえ」になるのは、(a)〜(e)のどれにも当てはまらない場合。典型は「〜て」「〜で」「〜けど」「〜から」「〜し」などの"
              "接続助詞・中止形で文がまだ続く形、")
F_NOTE_NEW = ("※「いいえ」になるのは、(a)〜(f)のどれにも当てはまらない場合。典型は「〜て」「〜で」「〜けど」「〜から」「〜し」などの"
              "接続助詞・中止形で終わり、同じ話者がその文の続きを話す形、")
F_CTX = ("※文末の形だけで決めず、後続の発話履歴で同じ話者がその文の続きを話しているかを確かめること。"
         "「ね」「これね」「それでね」のような前置き・つなぎや、「〜みたいな」「〜って感じ」の後に同じ話者が文を続けている場合は、"
         "(c) や (e) の形をしていても「いいえ」。")
OVL = re.compile(r"</?[^<>\n]{1,20}?と重複>")


def to_def(r):
    u = r["user"]
    for d in (REASON_DEF, HEAD_DEF):
        for k, v in d.items():
            assert k in u, k
            u = u.replace(k, v)
    return {**r, "system": SYSTEM_DEF, "user": u}


def add_e(r):
    s = r["system"]
    for k in (E_OLD, E_NOTE_OLD):
        assert k in s, k
    s = s.replace(E_OLD, E_NEW).replace(E_NOTE_OLD, E_NOTE_NEW)
    i = s.index("(d) ")
    j = s.index("\n", i)
    s = s[:j + 1] + E_ADD + "\n" + s[j + 1:]
    return {**r, "system": s}


def add_f(r):
    s = add_e(r)["system"]
    for k in (E_NEW, F_NOTE_OLD):
        assert k in s, k
    s = s.replace(E_NEW, F_NEW).replace(F_NOTE_OLD, F_NOTE_NEW)
    i = s.index("(e) ")
    j = s.index("\n", i)
    s = s[:j + 1] + F_ADD + "\n" + s[j + 1:]
    i = s.index(F_NOTE_NEW)
    j = s.index("\n", i)
    s = s[:j + 1] + F_CTX + "\n" + s[j + 1:]
    return {**r, "system": s}


def no_ovl(r):
    return {**r, "user": OVL.sub("", r["user"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="tt_analysis/gold1000_prompts.jsonl")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8")]
    rows = [r for r in rows if r["route"] == "llm"]
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for name, f in (("def", to_def), ("noovl", no_ovl), ("def_noovl", lambda r: no_ovl(to_def(r))),
                    ("add_e", add_e), ("add_f", add_f)):
        with open(out / f"gold1000_prompts.{name}.jsonl", "w", encoding="utf-8") as w:
            for r in rows:
                w.write(json.dumps(f(r), ensure_ascii=False) + "\n")
        n_ovl = sum(bool(OVL.search(f(r)["user"])) for r in rows)
        print(f"{name}: {len(rows)} 件（重なりタグが残る件数 {n_ovl}）")


if __name__ == "__main__":
    main()
