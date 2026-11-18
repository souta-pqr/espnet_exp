#!/usr/bin/env python
"""段階 2：result_oneshot.txt の LLM 判定区間（約 13.8 万件）を vLLM のモデルで付け直す。

設定は段階 1（run_gold.py）と同じ：chat template、貪欲デコード、_parse_response と同じ読み取り。
  <outdir>/<name>.tsv           LINE<TAB>はい/いいえ（読めなかった応答は いいえ）
  <outdir>/<name>.unparsed.tsv  LINE<TAB>生の応答（JSON 文字列。--thinking では末尾 500 字）。読めなかったものだけ
  <outdir>/<name>.ntok.tsv      LINE<TAB>出力トークン数<TAB>finish_reason（--thinking のときだけ）
チャンクごとに追記し、再開時は書き済みの LINE を飛ばす。生の応答は読めなかったもの以外は保存しない。

  python run_full.py --model Qwen/Qwen3-32B-AWQ --name Qwen3-32B-AWQ-nothink --max-model-len 4608 --gpu-mem 0.95
  python run_full.py --model Qwen/Qwen3-8B --name Qwen3-8B-think --thinking --max-tokens 4096 --max-model-len 10240
  python run_full.py --model Qwen/Qwen3-32B-AWQ --name Qwen3-32B-AWQ-nothink.silver13 --max-model-len 4608 \
      --reverse --other <outdir>/Qwen3-32B-AWQ-nothink.tsv   # 別のマシンの出力と分担し、後ろから進める
  python run_full.py ... --only-gold --outdir <dir>   # 本番の仕組みで gold の 922 件だけを出す（確認用）
"""
import argparse
import json
import os
import sys
import time

from full_items import GOLD, iter_llm_prompts, system_prompt
from run_gold import parse_response, final_text
from run_gold_prompts import final_text as final_text_v      # gemma / harmony 形式の思考の取り出し（2026-10-12）


def load_done(tsv, unp, ntok=None):
    """書き済みの LINE を返す。途中で切れた最終行は捨て、unparsed・ntok は tsv にある LINE だけに揃える。"""
    done = set()
    if os.path.exists(tsv):
        good = []
        for l in open(tsv, encoding="utf-8"):
            p = l.rstrip("\n").split("\t")
            if l.endswith("\n") and len(p) == 2 and p[0].isdigit() and p[1] in ("はい", "いいえ"):
                good.append(l); done.add(int(p[0]))
        with open(tsv, "w", encoding="utf-8") as f:
            f.writelines(good)
    for side in (unp, ntok):
        if side is None or not os.path.exists(side):
            continue
        keep = {}
        for l in open(side, encoding="utf-8"):
            p = l.rstrip("\n").split("\t", 1)
            if l.endswith("\n") and len(p) == 2 and p[0].isdigit() and int(p[0]) in done:
                keep[int(p[0])] = l
        with open(side, "w", encoding="utf-8") as f:
            f.writelines(keep[k] for k in sorted(keep))
    return done


def read_lines(p):
    """別のマシンが書いている tsv から、書き終わった行の LINE だけを読む（読むだけで書き換えない）。"""
    done = set()
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            q = l.rstrip("\n").split("\t")
            if l.endswith("\n") and len(q) == 2 and q[0].isdigit() and q[1] in ("はい", "いいえ"):
                done.add(int(q[0]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thinking", action="store_true", help="Qwen3 の enable_thinking=True")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--only-gold", action="store_true", help="gold1000 の llm 922 件だけ（確認用）")
    ap.add_argument("--reverse", action="store_true", help="後ろ（大きい LINE）から処理する")
    ap.add_argument("--other", help="分担相手の出力 tsv。チャンクごとに読み直し、そこにある LINE は飛ばす")
    ap.add_argument("--outdir", default="/autofs/diamond5/share/users/kobori/relabel/full")
    ap.add_argument("--family", default="qwen3", help="qwen3 / gemma / gpt-oss（思考部分の取り出し方）")
    ap.add_argument("--llm-kwargs", default="{}", help="vLLM の LLM() に足す引数（JSON）")
    ap.add_argument("--routes", default="llm", help="判定する経路（空白区切り．llm / rule_nextdiff）")
    ap.add_argument("--lines", help="この line_no の一覧（1 行 1 つ）だけ判定する（確認用）")
    ap.add_argument("--system", help="system プロンプトのファイル（既定は full_items.SYS。(e) 版などの差し替え用）")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    tsv = os.path.join(a.outdir, f"{a.name}.tsv")
    unp = os.path.join(a.outdir, f"{a.name}.unparsed.tsv")
    ntok = os.path.join(a.outdir, f"{a.name}.ntok.tsv") if a.thinking else None
    done = load_done(tsv, unp, ntok)

    t0 = time.time()
    items = list(iter_llm_prompts(routes=tuple(a.routes.split())))
    if a.only_gold:
        g = {json.loads(l)["line_no"] for l in open(GOLD, encoding="utf-8")}
        items = [x for x in items if x[0] in g]
    if a.lines:
        keep = {int(l) for l in open(a.lines) if l.strip()}
        items = [x for x in items if x[0] in keep]
    total = len(items)
    todo = [x for x in items if x[0] not in done]
    del items
    if a.reverse:
        todo.reverse()
    print(f"対象 {total} 件 / 済み {total - len(todo)} / 残り {len(todo)}（プロンプト作成 {time.time() - t0:.0f} 秒）",
          flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, **json.loads(a.llm_kwargs))
    tok = llm.get_tokenizer()
    print(f"モデル読み込み {time.time() - t0:.0f} 秒", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, skip_special_tokens=(a.family == "qwen3"))
    sys_p = open(a.system, encoding="utf-8").read().strip() if a.system else system_prompt()

    def render(user):
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=a.thinking)

    n_done = 0; n_bad = 0; t_run = 0.0; n_other = 0
    fn = open(ntok, "a", encoding="utf-8") if ntok else None
    with open(tsv, "a", encoding="utf-8") as ft, open(unp, "a", encoding="utf-8") as fu:
        while todo:
            if a.other:
                other = read_lines(a.other)
                n0 = len(todo)
                todo = [x for x in todo if x[0] not in other]
                n_other += n0 - len(todo)
                if not todo:
                    print(f"{time.strftime('%F %T')}  残りはすべて {a.other} にある → 終了", flush=True)
                    break
            ch = todo[:a.chunk]; todo = todo[a.chunk:]
            t1 = time.time()
            outs = llm.generate([render(u) for _, u in ch], sp, use_tqdm=False)
            dt = time.time() - t1
            lines = []; bad = []; toks = []
            for (ln, _), o in zip(ch, outs):
                raw = o.outputs[0].text
                txt = final_text(raw, "qwen3") if a.family == "qwen3" else final_text_v(raw, a.family)
                lab = parse_response(txt) if txt is not None else None
                lines.append(f"{ln}\t{'はい' if lab == 1 else 'いいえ'}\n")
                if lab not in (0, 1):
                    bad.append(f"{ln}\t{json.dumps(raw[-500:] if a.thinking else raw, ensure_ascii=False)}\n")
                toks.append(f"{ln}\t{len(o.outputs[0].token_ids)}\t{o.outputs[0].finish_reason}\n")
            fu.writelines(bad); fu.flush()          # unparsed・ntok を先に書く（tsv に無い LINE は再開時に捨てる）
            if fn:
                fn.writelines(toks); fn.flush()
            ft.writelines(lines); ft.flush()
            n_done += len(ch); n_bad += len(bad); t_run += dt
            spi = t_run / n_done
            rest = len(todo) * spi
            oth = f"  相手側で済み {n_other}" if a.other else ""
            print(f"{time.strftime('%F %T')}  {total - len(todo) - n_other}/{total}{oth}  "
                  f"{dt / len(ch):.3f} s/件（平均 {spi:.3f}）  読めず 累計 {n_bad}  "
                  f"残り {rest / 3600:.1f} 時間", flush=True)
    print(f"完了: 今回 {n_done} 件 / {t_run / 3600:.2f} 時間 / 読めず {n_bad}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
