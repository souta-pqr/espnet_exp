#!/usr/bin/env python
"""人手正解 1000 件のうち route=="llm" の 922 件を vLLM のモデルで判定し直す（段階 1）。

入力は gold1000_prompts.jsonl の system / user だけ（gold・qwen フィールドは見ない）。
貪欲デコード（temperature 0）。応答の読み取りは detect_section_text.py の _parse_response と同じ。
結果は チャンクごとに追記し、再実行すると済んだ line_no を飛ばす（途中で止められても再開できる）。

出力:
  <outdir>/<name>.tsv          line_no<TAB>はい/いいえ（読めなかった応答は _parse_response と同じく いいえ）
  <outdir>/<name>.raw.jsonl    line_no, 生の応答, 読み取り可否, 出力トークン数
  <outdir>/<name>.timing.jsonl チャンクごとの件数・経過秒

  python run_gold.py --model Qwen/Qwen3-8B --name Qwen3-8B-think --thinking
"""
import argparse
import json
import os
import re
import time

PROMPTS = "/autofs/diamond5/share/users/kobori/relabel/gold1000_prompts.jsonl"
OUTDIR = "/autofs/diamond5/share/users/kobori/relabel"


def parse_response(text):
    """detect_section_text.py の _parse_response と同じ規則。ただし読めなかったら None を返す。"""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [
        ln for ln in lines
        if '判定: はい または いいえ' not in ln
        and '以下の1行だけを出力して' not in ln
        and 'のいずれかで答えよ' not in ln
        and 'のどれかに当てはまれば' not in ln
    ]
    compact = '\n'.join(lines)
    m = re.search(r'^判定\s*[:：]\s*(はい|いいえ|相槌)\s*$', compact, flags=re.MULTILINE)
    if not m:
        m = re.search(r'^(はい|いいえ|相槌)\s*$', compact, flags=re.MULTILINE)
    if not m:
        for ln in lines[:3]:
            m2 = re.search(r'(はい|いいえ|相槌)', ln)
            if m2:
                m = m2
                break
    if m:
        word = m.group(1)
        return 1 if word == 'はい' else (2 if word == '相槌' else 0)
    return None


def final_text(raw, family):
    """思考部分を除いた最終応答。思考が閉じずに打ち切られたら None（＝読めなかった扱い）。"""
    if family == "gpt-oss":
        key = "<|channel|>final<|message|>"
        if key not in raw:
            return None
        t = raw.split(key, 1)[1]
        return re.sub(r"<\|[a-z_]+\|>", "", t)
    if "<think>" in raw and "</think>" not in raw:
        return None
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--thinking", action="store_true", help="Qwen3 の enable_thinking=True")
    ap.add_argument("--reasoning-effort", default="medium", help="gpt-oss の推論量")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="試し用に先頭 n 件だけ")
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()

    family = "gpt-oss" if "gpt-oss" in a.model.lower() else "qwen3"
    tsv = os.path.join(a.outdir, f"{a.name}.tsv")
    rawp = os.path.join(a.outdir, f"{a.name}.raw.jsonl")
    timp = os.path.join(a.outdir, f"{a.name}.timing.jsonl")

    items = []
    for l in open(PROMPTS, encoding="utf-8"):
        r = json.loads(l)
        if r["route"] == "llm":
            items.append({"line_no": r["line_no"], "system": r["system"], "user": r["user"]})
    if a.limit:
        items = items[:a.limit]
    done = set()
    if os.path.exists(tsv):
        done = {int(l.split("\t")[0]) for l in open(tsv, encoding="utf-8") if l.strip()}
    todo = [r for r in items if r["line_no"] not in done]
    print(f"対象 {len(items)} 件 / 済み {len(items) - len(todo)} / 残り {len(todo)}", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, trust_remote_code=False)
    tok = llm.get_tokenizer()
    print(f"モデル読み込み {time.time() - t0:.0f} 秒", flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens,
                        skip_special_tokens=(family != "gpt-oss"))

    def render(r):
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
        kw = {}
        if family == "qwen3":
            kw["enable_thinking"] = a.thinking
        else:
            kw["reasoning_effort"] = a.reasoning_effort
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    print("--- 描画例 ---\n" + render(todo[0])[:300] + "\n...\n" + render(todo[0])[-200:], flush=True)

    with open(tsv, "a", encoding="utf-8") as ft, open(rawp, "a", encoding="utf-8") as fr, \
            open(timp, "a", encoding="utf-8") as fm:
        for i in range(0, len(todo), a.chunk):
            ch = todo[i:i + a.chunk]
            t1 = time.time()
            outs = llm.generate([render(r) for r in ch], sp, use_tqdm=False)
            dt = time.time() - t1
            for r, o in zip(ch, outs):
                raw = o.outputs[0].text
                ft_ = final_text(raw, family)
                lab = parse_response(ft_) if ft_ is not None else None
                ok = lab in (0, 1)
                ft.write(f"{r['line_no']}\t{'はい' if lab == 1 else 'いいえ'}\n")
                fr.write(json.dumps({"line_no": r["line_no"], "parsed": ok, "label": lab,
                                     "finish": o.outputs[0].finish_reason,
                                     "n_out": len(o.outputs[0].token_ids),
                                     "n_in": len(o.prompt_token_ids), "raw": raw},
                                    ensure_ascii=False) + "\n")
            fm.write(json.dumps({"n": len(ch), "sec": dt}) + "\n")
            for f in (ft, fr, fm):
                f.flush()
            print(f"{i + len(ch)}/{len(todo)}  {dt:.1f}s  ({dt / len(ch):.3f} s/件)", flush=True)


if __name__ == "__main__":
    main()
