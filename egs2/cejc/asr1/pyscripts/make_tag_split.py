#!/usr/bin/env python3
"""text ファイルをタグと発話に分割するスクリプト。

入力 (data/<split>/text):
    uttid <no>    ふ ー ん
    uttid <yes>   は い
    uttid <other> あ の で

出力:
    data/<split>/text.with_tag : バックアップ (元の text のコピー、変更なし)
    data/<split>/tag           : タグのみ  ("uttid <no>")
    data/<split>/text          : 発話のみに上書き  ("uttid ふ ー ん")

使い方:
    python pyscripts/make_tag_split.py data/train_nodup data/train_dev data/eval
    python pyscripts/make_tag_split.py data/train_nodup  # 1 データセットだけ
    python pyscripts/make_tag_split.py data/train_nodup --dry-run  # 確認のみ
"""
import argparse
import os
import shutil
import sys
from collections import Counter


TAG_TOKENS = {"<no>", "<yes>", "<other>"}
TAG_TO_INT = {"<no>": "0", "<yes>": "1", "<other>": "2"}


def split_text(data_dir: str, dry_run: bool = False) -> None:
    text_path = os.path.join(data_dir, "text")
    tag_path = os.path.join(data_dir, "tag")
    backup_path = os.path.join(data_dir, "text.with_tag")

    if not os.path.isfile(text_path):
        print(f"[SKIP] {text_path} が見つかりません。", file=sys.stderr)
        return

    # バックアップが既に存在する場合は分割済みとみなす
    if os.path.isfile(backup_path):
        print(f"[SKIP] {data_dir} は既に分割済みです (text.with_tag が存在)。")
        print(f"       再実行したい場合は text.with_tag を削除してください。")
        return

    # 元の text を読む
    with open(text_path, encoding="utf-8") as f:
        lines = f.readlines()

    tag_lines = []
    text_lines = []  # 発話のみ (text に上書き)
    n_no_tag = 0

    for line in lines:
        parts = line.rstrip("\n").split()
        if not parts:
            continue
        uttid = parts[0]

        if len(parts) >= 2 and parts[1] in TAG_TOKENS:
            tag = parts[1]
            content_parts = parts[2:]
        else:
            tag = "<other>"
            content_parts = parts[1:]
            n_no_tag += 1

        content = " ".join(content_parts)
        tag_int = TAG_TO_INT.get(tag, "2")
        tag_lines.append(f"{uttid} {tag_int}\n")
        text_lines.append(f"{uttid} {content}\n")

    if n_no_tag > 0:
        print(f"  warning: タグが見つからない発話が {n_no_tag} 件あります。<other> で補完しました。")

    if dry_run:
        print(f"  [dry-run] backup  → {backup_path}")
        print(f"  [dry-run] tag     → {tag_path}    ({len(tag_lines)} utterances)")
        print(f"  [dry-run] text    ← 発話のみに上書き ({len(text_lines)} utterances)")
    else:
        # 1. バックアップ (text → text.with_tag)
        shutil.copy(text_path, backup_path)
        print(f"  backup  → {backup_path}")

        # 2. tag ファイル作成
        with open(tag_path, "w", encoding="utf-8") as f:
            f.writelines(tag_lines)
        print(f"  tag     → {tag_path}    ({len(tag_lines)} utterances)")

        # 3. text を発話のみに上書き
        with open(text_path, "w", encoding="utf-8") as f:
            f.writelines(text_lines)
        print(f"  text    ← 発話のみに上書き ({len(text_lines)} utterances)")

    # タグ分布
    INT_TO_TAG = {v: k for k, v in TAG_TO_INT.items()}
    tag_counts = Counter(line.split()[1] for line in tag_lines if len(line.split()) >= 2)
    for int_val, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        tag_name = INT_TO_TAG.get(int_val, int_val)
        print(f"    {tag_name} ({int_val}): {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dirs",
        nargs="+",
        help="処理するデータディレクトリ (data/train_nodup など)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="ファイルを書き出さずに内容のみ確認する",
    )
    args = parser.parse_args()

    for data_dir in args.data_dirs:
        print(f"\n[{data_dir}]")
        split_text(data_dir, dry_run=args.dry_run)

    print("\n完了。")
    if not args.dry_run:
        print("次のステップ: bash run_cif_transformer.sh --stage 4 --stop_stage 4 でデータを dump に再コピーしてください。")


if __name__ == "__main__":
    main()
