# 発話区間末ラベルの付与（Qwen3-8B oneshot）

学習・評価のラベル（`data/*_g/tag`。0=継続 / 1=完了 / 2=相槌）を作った手順。
元は `~/work/cejc-label/`（2026-10-05 にユーザが配置）。ここにはその写しを置く。

## 判定の順序（`detect_section_text.py` の `process()`）

1. **相槌（2）**: フィラー等を除いた本文が相槌の一覧（`AIZUCHI_SET`）にあるか、空になる → 規則で 2
2. **完了（1）**: 判定対象の後に最初に話し始めるのが別の話者 → 規則で 1
3. **それ以外**: Qwen3-8B（`enable_thinking=False`、`do_sample=False`）が「はい（1）／いいえ（0）」を答える
   - プロンプトは `oneshot/prompt_system.txt` と `oneshot/prompt_user.txt`（直前 10 発話＋判定対象。例 2 つ）
4. 各会話の先頭 `context_n`（10）・末尾 `post_context_n`（10）発話は対象外（ラベルなし）

## 確認したこと（2026-10-05）

ラベルの元 `/autofs/diamond5/share/users/kobori/result_oneshot.txt`（397,024 件）に上の規則を当て直すと、
相槌の規則 110,251 件はすべて 2、別話者の規則 148,961 件はすべて 1、残り 137,812 件が LLM の判定（0: 71,564 / 1: 66,248）で、
規則の部分は完全に一致した。LLM の部分は再実行していない。

## 実行例

```bash
python local/label/detect_section_text.py \
    --text_file <時系列テキスト（開始=… 終了=… 話者=… 発話=…、会話ごとに空行区切り）> \
    --result_file <出力> --model Qwen3-8B \
    --prompt_file local/label/oneshot/prompt_user.txt \
    --system_prompt_file local/label/oneshot/prompt_system.txt \
    --context_n 10 --post_context_n 10 --batch_size 8
```

出力の `[LINE n] 会話=… 発話位置=…` の番号は入力ファイルの並びに依存する。
**発話 ID への対応づけは番号ではなく時刻で行うこと**（`local/turntaking/gold1000_map.py`）。
