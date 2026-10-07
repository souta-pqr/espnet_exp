#!/usr/bin/env python
"""ASR 認識結果から BERT の表現を取り出し、kaldi ark+scp に書き出す。

音声ヘッドに渡して共同学習するための入力。後段で確率を平均する方式（+0.020）に対し、
表現の段階で混ぜれば相互作用を使える。

**因果性**：ここで作るのは「発話全体の認識結果」の表現なので、
発話の途中では手に入らない。したがってこのベクトルを使ってよいのは
**音声区間検出が発火して認識結果が確定した後の判定**だけである。
それより早い時点の判定は音声のみで行う（モデル側でマスクする）。

  python local/turntaking/dump_text_vec.py --dset eval --text_scp <ASR仮説>
"""
import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_ceiling import build                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dset", required=True)
    ap.add_argument("--text_scp", required=True, help="ASR 認識結果の text")
    ap.add_argument("--bert", default="exp/text_bert_asr")
    ap.add_argument("--out", default="")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    import kaldiio
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    rows = build(args.dset, text_scp=args.text_scp)
    # ESPnet は各入力のキー順が揃っていることを要求するので utt id 順に並べる
    rows.sort(key=lambda r: r["utt"])
    print(f"{args.dset}: {len(rows)} 区間", flush=True)
    tk = AutoTokenizer.from_pretrained(args.bert)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.bert, output_hidden_states=True).cuda().eval()

    class DS(Dataset):
        def __init__(self, r): self.r = r
        def __len__(self): return len(self.r)
        def __getitem__(self, i): return self.r[i]

    def collate(b):
        return (tk([x["text"] for x in b], truncation=True, max_length=args.max_len,
                   padding=True, return_tensors="pt"), [x["utt"] for x in b])

    out = args.out or f"dump/raw/{args.dset}/text_vec"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with kaldiio.WriteHelper(f"ark,scp:{out}.ark,{out}.scp") as w, torch.no_grad():
        for enc, utts in DataLoader(DS(rows), batch_size=args.bs, collate_fn=collate,
                                    num_workers=4):
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model(**enc).hidden_states[-1][:, 0]        # CLS
            h = h.float().cpu().numpy()
            for u, v in zip(utts, h):
                w(u, v.astype(np.float32))
            n += len(utts)
            if n % 20000 == 0:
                print(f"  ...{n}", flush=True)
    print(f"書き出し {n} 件 → {out}.scp（{h.shape[1]} 次元）")


if __name__ == "__main__":
    main()
