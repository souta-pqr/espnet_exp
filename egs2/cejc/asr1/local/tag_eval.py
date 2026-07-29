#!/usr/bin/env python3
"""CIF TEASER タグ予測評価スクリプト

評価指標:
  - Accuracy（全体・クラスごと）
  - Macro / Per-class F1
  - Earliness: 確定した fire / (総 fire 数 - 1)  0=最初, 1=最後

使用例:
    python local/tag_eval.py \
        --config  exp/asr_0522-cif-tag-transformer-cejc-epoch150-v1/config.yaml \
        --model   exp/asr_0522-cif-tag-transformer-cejc-epoch150-v1/valid.cer.best.pth \
        --wav_scp dump/raw/eval/wav.scp \
        --tag     dump/raw/eval/tag \
        --output  exp/asr_0522-cif-tag-transformer-cejc-epoch150-v1/tag_eval_result.tsv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ESPnet が ~/2025/espnet 配下にある場合のパス補完
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from espnet2.tasks.asr import CIFASRTask

CLASS_NAMES = ["<no>", "<yes>", "<other>"]


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def load_scp(path, base_dir=None):
    """key-value 形式のファイルを読む。値が相対パスなら base_dir で解決する。"""
    data = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(None, 1)
            if base_dir and not Path(val).is_absolute():
                val = str(Path(base_dir) / val)
            data[key] = val
    return data


def f1_per_class(y_true, y_pred, n_classes):
    """各クラスの precision / recall / F1 と macro 平均を返す。"""
    results = {}
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[c] = {"precision": prec, "recall": rec, "f1": f1,
                      "support": (y_true == c).sum()}
    macro_f1 = np.mean([results[c]["f1"] for c in range(n_classes)])
    return results, macro_f1


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CIF タグ早期確定 評価（SPRT / TEASER）")
    parser.add_argument("--config",  required=True, help="config.yaml のパス")
    parser.add_argument("--model",   required=True, help=".pth モデルファイルのパス")
    parser.add_argument("--wav_scp", required=True, help="dump/raw/eval/wav.scp")
    parser.add_argument("--tag",     required=True, help="dump/raw/eval/tag（正解ラベル）")
    parser.add_argument("--output",  default=None,  help="詳細結果の保存先 .tsv（省略可）")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--base_dir", default=".",
                        help="wav.scp の相対パスを解決する基準ディレクトリ（デフォルト: カレント）")
    # --- 手法選択 ---
    parser.add_argument("--method", default="sprt", choices=["sprt", "teaser"],
                        help="評価手法: sprt（手法C, デフォルト）or teaser（手法A）")
    # --- SPRT パラメータ ---
    parser.add_argument("--streaming", action="store_true",
                        help="SPRT ストリーミング推論を使用する（--method sprt のみ）")
    parser.add_argument("--chunk_frames", type=int, default=76,
                        help="ストリーミング時のチャンクサイズ（特徴量フレーム数, デフォルト: 76）")
    parser.add_argument("--sprt_upper", type=float, default=None,
                        help="SPRT 停止境界（省略時はモデルの sprt_upper を使用）")
    # --- TEASER パラメータ ---
    parser.add_argument("--stop_threshold", type=float, default=None,
                        help="TEASER: stop_head の確率閾値（省略時はモデルの stop_threshold を使用）")
    parser.add_argument("--teaser_v", type=int, default=None,
                        help="TEASER: 同クラス連続確定回数（省略時はモデルの teaser_v を使用）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    device = torch.device(args.device)

    # --- モデルロード ---
    logging.info(f"モデルをロード中: {args.model}")
    model, _ = CIFASRTask.build_model_from_file(args.config, args.model, args.device)
    model.eval()
    logging.info("モデルロード完了")

    # --- データロード ---
    wav_scp = load_scp(args.wav_scp, base_dir=args.base_dir)
    tag_scp = load_scp(args.tag)

    uttids = sorted(set(wav_scp.keys()) & set(tag_scp.keys()))
    logging.info(f"評価発話数: {len(uttids)}")

    y_true        = []
    y_pred        = []
    earliness_all = []
    details       = []   # (uttid, gt, pred, dec_fire, n_fires, earliness)

    with torch.no_grad():
        for idx, uttid in enumerate(uttids):
            # --- 音声ロード ---
            wav_path = wav_scp[uttid]
            try:
                speech_np, sr = sf.read(wav_path, dtype="float32")
            except Exception as e:
                logging.warning(f"スキップ ({uttid}): {e}")
                continue

            # モノラル化
            if speech_np.ndim == 2:
                speech_np = speech_np.mean(axis=1)

            speech = torch.tensor(speech_np).unsqueeze(0).to(device)          # (1, T)
            speech_lengths = torch.tensor([speech.shape[1]], dtype=torch.long).to(device)

            # --- 推論（手法ごとに分岐）---
            try:
                if args.method == "teaser":
                    tag_pred, decision_fire, n_fires = model.inference_tag_teaser(
                        speech, speech_lengths,
                        stop_threshold=args.stop_threshold,
                        v=args.teaser_v,
                    )
                    pred   = tag_pred[0].item()
                    d_fire = decision_fire[0].item()
                    n_fire = n_fires[0].item()
                elif args.streaming:
                    pred_i, d_fire_i, n_fire_i = model.inference_tag_streaming(
                        speech, speech_lengths,
                        sim_chunk_feat_frames=args.chunk_frames,
                        sprt_upper=args.sprt_upper,
                    )
                    pred   = pred_i
                    d_fire = d_fire_i
                    n_fire = n_fire_i
                else:
                    tag_pred, _, _, decision_fire, n_fires = \
                        model.inference_tag_with_earliness(
                            speech, speech_lengths,
                            sprt_upper=args.sprt_upper,
                        )
                    pred   = tag_pred[0].item()
                    d_fire = decision_fire[0].item()
                    n_fire = n_fires[0].item()
            except Exception as e:
                logging.warning(f"推論失敗 ({uttid}): {e}")
                continue

            gt    = int(tag_scp[uttid].strip())
            # earliness: 0 = 最初の fire で確定、1 = 最後の fire まで待った
            early = d_fire / max(n_fire - 1, 1)

            y_true.append(gt)
            y_pred.append(pred)
            earliness_all.append(early)
            details.append((uttid, gt, pred, d_fire, n_fire, early))

            if (idx + 1) % 1000 == 0:
                logging.info(f"  {idx + 1}/{len(uttids)} 件完了")

    if len(y_true) == 0:
        logging.error("有効な評価データがありませんでした")
        return

    y_true        = np.array(y_true)
    y_pred        = np.array(y_pred)
    earliness_arr = np.array(earliness_all)

    # ---------------------------------------------------------------------------
    # 結果出力
    # ---------------------------------------------------------------------------
    sep = "=" * 60

    print(f"\n{sep}")
    print("【タグ分類 Accuracy】")
    print(sep)
    acc = (y_true == y_pred).mean()
    print(f"  全体 Accuracy : {acc:.3f}  ({(y_true == y_pred).sum()} / {len(y_true)})")
    print()
    for c, name in enumerate(CLASS_NAMES):
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        c_acc = (y_pred[mask] == c).mean()
        print(f"  {name:10s}: {c_acc:.3f}  (正解数 {(y_pred[mask]==c).sum()} / {mask.sum()} サンプル)")

    print(f"\n{sep}")
    print("【F1 スコア】")
    print(sep)
    per_class, macro_f1 = f1_per_class(y_true, y_pred, len(CLASS_NAMES))
    print(f"  {'クラス':12s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>8s}  {'Support':>8s}")
    print(f"  {'-'*55}")
    for c, name in enumerate(CLASS_NAMES):
        r = per_class[c]
        print(f"  {name:12s}  {r['precision']:10.3f}  {r['recall']:8.3f}  {r['f1']:8.3f}  {r['support']:8d}")
    print(f"  {'-'*55}")
    print(f"  {'Macro avg':12s}  {'':10s}  {'':8s}  {macro_f1:8.3f}  {len(y_true):8d}")

    print(f"\n{sep}")
    print("【混同行列】（行=正解、列=予測）")
    print(sep)
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for gt, pr in zip(y_true, y_pred):
        cm[gt][pr] += 1
    header = f"{'':12s}" + "".join(f"{n:>10s}" for n in CLASS_NAMES)
    print(f"{'':12s}  予測 →")
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:12s}" + "".join(f"{cm[i, j]:10d}" for j in range(len(CLASS_NAMES)))
        print(row)

    print(f"\n{sep}")
    print("【Earliness（早期確定指標）】")
    print(sep)
    print(f"  平均 Earliness  : {earliness_arr.mean():.3f}")
    print(f"  中央値          : {np.median(earliness_arr):.3f}")
    print(f"  標準偏差        : {earliness_arr.std():.3f}")
    print()

    # HM: accuracy と (1 - earliness) の調和平均（sprt_upper sweep の選択指標）
    mean_early = earliness_arr.mean()
    earliness_score = 1.0 - mean_early
    if acc + earliness_score > 0:
        hm = 2 * acc * earliness_score / (acc + earliness_score)
    else:
        hm = 0.0
    print(f"  HM (acc × (1-earliness)): {hm:.3f}  "
          f"[acc={acc:.3f}, 1-earliness={earliness_score:.3f}]")
    print()
    buckets = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels  = ["0〜25%", "25〜50%", "50〜75%", "75〜100%", "最後"]
    print(f"  確定位置の分布 (全 fire を 1 として):")
    prev = -1
    for thresh, label in zip(buckets, labels):
        if thresh == 1.0:
            cnt = (earliness_arr == 1.0).sum()
            print(f"    最後の fire のみ: {cnt:6d} 件  ({100*cnt/len(earliness_arr):5.1f}%)")
        else:
            cnt = ((earliness_arr > prev) & (earliness_arr <= thresh)).sum()
            print(f"    {label:10s}      : {cnt:6d} 件  ({100*cnt/len(earliness_arr):5.1f}%)")
        prev = thresh

    print()
    print(f"  クラス別平均 Earliness:")
    for c, name in enumerate(CLASS_NAMES):
        mask = (y_true == c)
        if mask.sum() > 0:
            print(f"    {name:10s}: {earliness_arr[mask].mean():.3f}  (n={mask.sum()})")

    # ---------------------------------------------------------------------------
    # 詳細結果 TSV 保存
    # ---------------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("uttid\tgt\tpred\tcorrect\tdecision_fire\tn_fires\tearlyness\n")
            for uttid, gt, pred, df, nf, e in details:
                correct = int(gt == pred)
                f.write(f"{uttid}\t{gt}\t{pred}\t{correct}\t{df}\t{nf}\t{e:.4f}\n")
        logging.info(f"詳細結果を保存しました: {out_path}")


if __name__ == "__main__":
    main()
