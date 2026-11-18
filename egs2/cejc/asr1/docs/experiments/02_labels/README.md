# 02 ラベル

各区間の区間末に，継続（0）・完了（1）・相槌（2）のラベルを付ける（`data/*/tag`）．
判定するのは「伝えたいことを言い終えたか」という意味で，無音の長さや次に誰が話すかではない．

[← 全体の README](../../../../../../README.md)

## 今のラベル（gemma-4-31B）

| 区間 | 付け方 |
| --- | --- |
| 相槌の語の一覧（`local/label/detect_section_text.py` の `AIZUCHI_SET`）と一致する区間，一覧の語を並べただけの区間 | 規則で相槌 |
| それ以外 | gemma-4-31B（思考あり，(e) 版のプロンプト）が「言い終えたか」を判定 |

人手正解 1000 件との一致率は 0.874（その前の 3 モデルの多数決＋規則は 0.832）．

## 手順

### 1. LLM で判定する（silver13・GPU）

スクリプトは `local/label/relabel_llm/`．vLLM を使う（conda 環境 `llm`）．モデルは研究室の共有キャッシュ（`/data/huggingface/hub`）から読む．

```bash
cd local/label/relabel_llm
python full_items.py --check                       # 判定する区間とプロンプトを作り，件数を確かめる
bash run_full_g31.sh                               # LLM が判定する区間（137,812 件．約 76 時間）
V=add_e ROUTES=rule_nextdiff bash run_full_route.sh   # 「直後に別の話者なら完了」にしていた区間（148,961 件．約 82 時間）
```

- 止めても続きから再開できる（出力済みの区間を飛ばす）．`setsid nohup` で起動する．
- 出力は `/autofs/diamond5/share/users/kobori/relabel/full_e/`・`full_route/`．
- 元のプロンプトと規則は [local/label/README.md](../../../local/label/README.md)．(e) は「普通の言い切り・体言止め・倒置・短い応答も言い終えている」を足した基準．

### 2. プロンプトやモデルを変えるときは，先に人手正解で確かめる

```bash
# local/label/relabel_llm で（silver13・GPU）
python run_gold.py --model Qwen/Qwen3-8B --name Qwen3-8B-think --thinking   # 人手正解 922 件だけを判定
python run_gold_prompts.py ...    # プロンプトの変種で判定（変種は local/turntaking/gold1000_prompt_variants.py．例は run_prompt_check*.sh）

# egs2/cejc/asr1 で
python local/turntaking/gold1000_prompt_compare.py   # 一致率・文末の形ごとの一致率を比べる
python local/turntaking/gold1000_errors.py           # 人手と食い違う区間の傾向
```

### 3. `data/*/tag` を書き換える

```bash
python local/turntaking/gemma_tags.py           # 集計だけ（何も書かない）
python local/turntaking/gemma_tags.py --write   # data/<set>/tag を tag.vote_e に退避してから書き換える
```

`tt_batch_logs/run_g_stage2gm.sh` の段 1 が，`dump/raw/*/tag` と `train_gx` の tag も同じように書き換える．

## ほかのラベル（前の版）

- 3 モデル（Qwen3-8B・Qwen3-32B・Qwen3-8B 思考あり）の多数決：`local/turntaking/vote_tags.py`（記録は `docs/label_vote_20261007.md`）．
- ラベルの質の確認：`desumasu_check.py`（です・ますで終わる区間），`aizuchi_concat_check.py`（相槌の語の並び），`compare_route_check.py`（規則で完了にしていた区間），`gold_add_sheet.py`（人手正解を足すための用紙）．すべて `local/turntaking/` にある．

## 注意

- LLM の判定結果と人手正解は CEJC の書き起こしを含むので，git に入れない．
