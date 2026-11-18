#!/usr/bin/env python3
"""段階 2（silver13）のジョブが終わった（または止まった）あとに relabel/full/README_silver13.md を書く。

  python finalize_readme.py <状態>   # 状態: done / gate_failed / error ... / 途中経過
"""
import os
import re
import statistics
import sys
import time

D = "/autofs/diamond5/share/users/kobori/relabel/full"
NAME = "Qwen3-8B-think"
L = "/home/kobori/relabel_llm/logs"
TOTAL = 137812


def lines(p):
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def text(p):
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


state = sys.argv[1] if len(sys.argv) > 1 else "途中経過"
tsv = lines(f"{D}/{NAME}.tsv")
n = len(tsv); yes = sum(l.endswith("\tはい") for l in tsv)
bad = len(lines(f"{D}/{NAME}.unparsed.tsv"))
ntok = [l.split("\t") for l in lines(f"{D}/{NAME}.ntok.tsv")]
nout = [int(x[1]) for x in ntok]
n_len = sum(x[2] == "length" for x in ntok)
guard = text(f"{L}/full_main.guard.log")
prog = [l for l in text(f"{L}/full_main.log").splitlines() if re.match(r"^\d{4}-\d\d-\d\d .*/%d " % TOTAL, l)]
starts = re.findall(r"^(\S+ \S+) 起動", guard, flags=re.M)
ends = re.findall(r"^(\S+ \S+) (ジョブ終了.*|他ユーザーの.*|停止完了)$", guard, flags=re.M)
last_spi = re.search(r"平均 ([\d.]+)", prog[-1]).group(1) if prog else "?"
check1 = text(f"{L}/check1_prompts.log").strip() or "（未実行）"
check2 = text(f"{L}/check2_gold.log").strip() or "（未実行）"
if state == "done" and n == TOTAL:
    status = "完了"
elif state == "gate_failed":
    status = f"起動前の確認で停止（本番 {n}/{TOTAL}）"
else:
    status = f"未完了（{n}/{TOTAL}、状態: {state}）"
tok = (f"平均 {statistics.mean(nout):.0f} / 中央値 {statistics.median(nout):.0f} / 最大 {max(nout)}"
       if nout else "?")

md = f"""# 段階 2：Qwen3-8B 思考ありによる全件付け直し（silver13）

状態: **{status}**（{time.strftime('%F %T')} 時点で自動作成）

## 件数
- result_oneshot.txt の [LINE]: 397,024 件
  - 相槌のルール（is_aizuchi）: 110,251 件 → 対象外
  - 完了のルール（is_turn_end_next_different_speaker）: 148,961 件 → 対象外
  - **LLM 判定の対象: {TOTAL:,} 件**
- 書き出し済み: {n:,} 件（はい {yes:,} / いいえ {n - yes:,}）
- **読めなかった応答: {bad} 件**（「いいえ」にした。うち思考が閉じずに max_tokens で打ち切られたもの {n_len} 件。LINE と応答の末尾 500 字は {NAME}.unparsed.tsv）
- 出力トークン数（思考を含む）: {tok}（件ごとの値は {NAME}.ntok.tsv：LINE<TAB>トークン数<TAB>finish_reason）

## 所要時間
- 起動: {', '.join(starts) or '?'}
- 終了・停止: {'; '.join(' '.join(e) for e in ends) or '?'}
- 1 件あたり（チャンク 512 件、prefix caching あり、モデル読み込み除く）: 平均 {last_spi} 秒

## 起動前の確認
1. 対象数とプロンプトの一致（full_items.py --check）
```
{check1}
```
2. 本番の仕組み（run_full.py --only-gold）で 922 件を出し直し、段階 1 の Qwen3-8B-think.tsv と比較（check_gold922/）
   - 当初の基準「違い 10 件以下」では 121 件違って止まった（2026-10-07 20:04）。思考ありは貪欲デコードでも出し直すたびに思考が分かれる（同じ GPU で 2 回回しても 99 件違った）ためで、入力の食い違いではなかった（check_gold922/think_rerun_variance.md）。
   - 小堀さんの了承を得て、基準を「出し直した結果でも 3 モデル多数決の改善の 95% 区間の下限 > 0」に置き換えて本番に入った。
```
{check2}
```

## 設定
- モデル: Qwen/Qwen3-8B（bf16）、vLLM 0.31.0（conda 環境 `llm`、torch 2.13.0+cu130）、RTX 5090 32GB。max_model_len 10240、gpu_memory_utilization 0.85、prefix caching あり。
- chat template（enable_thinking=True）、貪欲デコード（temperature 0、max_tokens 4096）。応答の読み取りは detect_section_text.py の _parse_response と同じ規則（思考部分を除いてから読む。思考が閉じていなければ読めなかった扱い）。
- 入力: full_context/oneshot の prompt_system.txt / prompt_user.txt を、LLMDetector.make_prompt で描画した。
- 出力: {NAME}.tsv（LINE番号<TAB>はい/いいえ）。思考の全文は保存していない。発話 ID への対応は relabel/qwen3_fc_os_uttid.tsv。
- 起動: cron（5 分おき）の watch.sh が、GPU に他ユーザーのプロセスが無い状態が 15 分続いたら、guard.py の下で切り離して起動する。他のユーザーの GPU プロセスが現れたら止まり、次に空いたら書き済みの LINE を飛ばして再開する。終わると cron エントリを自分で消す。
- スクリプト: relabel/scripts/silver13/（silver9 の full_items.py・run_full.py・run_gold.py・guard.py を元に、run_full.py に --thinking と出力トークン数の記録を足した）
"""
open(f"{D}/README_silver13.md", "w", encoding="utf-8").write(md)
print(md)
