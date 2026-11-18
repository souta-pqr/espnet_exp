#!/usr/bin/env python3
"""32B 思考なしの分担（silver13 が後ろから）の進み具合を relabel/full/README_silver13_32b.md に書く。

  python finalize_readme_32b.py <状態>   # 状態: done / gate_failed / error ... / 途中経過
"""
import os
import re
import sys
import time

D = "/autofs/diamond5/share/users/kobori/relabel/full"
MINE = f"{D}/Qwen3-32B-AWQ-nothink.silver13.tsv"
THEIRS = f"{D}/Qwen3-32B-AWQ-nothink.tsv"
L = "/home/kobori/relabel_llm/logs32"
TOTAL = 137812


def read(p):
    out = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            q = l.rstrip("\n").split("\t")
            if l.endswith("\n") and len(q) == 2 and q[0].isdigit():
                out[int(q[0])] = q[1]
    return out


def text(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


state = sys.argv[1] if len(sys.argv) > 1 else "途中経過"
mine, theirs = read(MINE), read(THEIRS)
both = set(mine) & set(theirs)
disagree = sum(mine[k] != theirs[k] for k in both)
union = len(set(mine) | set(theirs))
bad = len([l for l in text(f"{D}/Qwen3-32B-AWQ-nothink.silver13.unparsed.tsv").splitlines() if l.strip()])
yes = sum(v == "はい" for v in mine.values())
log = text(f"{L}/run32.log")
prog = [l for l in log.splitlines() if re.match(r"^\d{4}-\d\d-\d\d .*/%d " % TOTAL, l)]
spi = re.search(r"平均 ([\d.]+)", prog[-1]).group(1) if prog else "?"
guard = text(f"{L}/run32.guard.log")
starts = re.findall(r"^(\S+ \S+) 起動", guard, flags=re.M)
ends = re.findall(r"^(\S+ \S+) (ジョブ終了.*|他ユーザーの.*|停止完了)$", guard, flags=re.M)
gate = text(f"{L}/check_gold.log").strip() or "（未実行）"
lo = min(mine) if mine else None
hi_theirs = max(theirs) if theirs else None
if union == TOTAL:
    status = "完了"
elif state == "gate_failed":
    status = "起動前の確認で停止"
else:
    status = f"未完了（状態: {state}）"

md = f"""# 段階 2：Qwen3-32B-AWQ 思考なしの全件付け直し — silver13 の後ろからの分担

状態: **{status}**（{time.strftime('%F %T')} 時点で自動作成）

## 件数（LLM 判定の対象 {TOTAL:,} 件）
- **2 台を合わせた件数: {union:,} / {TOTAL:,}**（残り {TOTAL - union:,} 件）
  - silver9（前から、Qwen3-32B-AWQ-nothink.tsv）: {len(theirs):,} 件（最大の LINE {hi_theirs}）
  - silver13（後ろから、Qwen3-32B-AWQ-nothink.silver13.tsv）: {len(mine):,} 件（最小の LINE {lo}、はい {yes:,} / いいえ {len(mine) - yes:,}）
  - 2 台が両方書いた LINE: {len(both)} 件（うち判定が違う {disagree} 件）。合わせるときは silver9 の判定を使う。
- silver13 の読めなかった応答: **{bad} 件**（「いいえ」にした。LINE と生の応答は Qwen3-32B-AWQ-nothink.silver13.unparsed.tsv）

## 所要時間（silver13）
- 起動: {', '.join(starts) or '?'}
- 終了・停止: {'; '.join(' '.join(e) for e in ends) or '?'}
- 1 件あたり（チャンク 256 件、prefix caching あり、モデル読み込み除く）: 平均 {spi} 秒

## 起動前の確認
人手正解の 922 件（gold1000_prompts.jsonl の route=="llm"）を同じ設定で出し直し、段階 1 の relabel/Qwen3-32B-AWQ-nothink.tsv と比べた（check_gold922/Qwen3-32B-AWQ-nothink.silver13.*）。
```
{gate}
```

## 設定（silver9 の段階 2 と同じ）
- モデル: Qwen/Qwen3-32B-AWQ（revision 0499c3a）、vLLM 0.31.0（conda 環境 `llm`）、RTX 5090 32GB。max_model_len 5120（silver9 と同じ。全対象のプロンプトの最長は 4,931 トークン）、gpu_memory_utilization 0.85（silver9 は 3090 で 0.95）。最初は 4608 で起動し、10/9 01:14 に 4,609 トークンを超えるプロンプト（LINE 204691 など）で止まったので、silver9 に合わせて 5120 に上げて続きから再開した、prefix caching あり。
- chat template（enable_thinking=False）、貪欲デコード（temperature 0、max_tokens 64）。応答の読み取りは detect_section_text.py の _parse_response と同じ規則。
- 対象とプロンプト: relabel/scripts の full_items.py（silver9 と同じ）。
- 分担: 対象を後ろ（大きい LINE）から処理する。256 件ごとに silver9 の Qwen3-32B-AWQ-nothink.tsv を読み直し（読むだけ）、そこにある LINE は飛ばす。残りがすべて silver9 側で済んでいたら終わる。
- 起動: cron（5 分おき）の watch32.sh が、GPU に他ユーザーのプロセスが無い状態が 15 分続いたら、guard.py の下で切り離して起動する。他のユーザーの GPU プロセスが現れたら止まり、空いたら書き済みの LINE を飛ばして再開する。
- スクリプト: relabel/scripts/silver13/（run_full.py に --reverse と --other を足した。run32_main.sh, compare_gold32.py, watch32.sh, finalize_readme_32b.py）
"""
open(f"{D}/README_silver13_32b.md", "w", encoding="utf-8").write(md)
print(md)
