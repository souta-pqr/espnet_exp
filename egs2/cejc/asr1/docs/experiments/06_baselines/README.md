# 06 比較手法

提案手法と比べる方式を用意する．どれも公開の重みから続けて，提案手法と同じデータ（学習 `data/train_gx`，検証 `data/dev_g`）と同じラベルで学習し直す．
[04 エンコーダ固定＋Transformer](../04_xfmr/README.md) も比較手法の 1 つ．

[← 全体の README](../../../../../../README.md)

| 方式 | 学習し直すコード | 実時間の `--model` | 今の報告で使うか |
| --- | --- | --- | --- |
| Easy Turn（ASLP-lab，Apache-2.0） | `local/encac/train_et.py` | `easy_turn_ft` | 使う |
| エンコーダ固定＋Transformer | [04](../04_xfmr/README.md) | `proposed_new --no_past` | 使う |
| silero-vad のみ（無音の長さだけ） | 学習なし | `silero` | waseda_soma で使う |
| Smart Turn v3.2（pipecat-ai，BSD-2-Clause） | `local/baselines/smartturn_ft/` | `smart_turn_ft` | 前の報告まで |
| VAP（maai，CC BY-NC-ND 4.0） | `local/baselines/vap_ft/` | `vap_mono_ft` | 前の報告まで |

## Easy Turn

silver13・conda 環境 `easyturn`．Easy Turn のコード（`~/2026/work/easy_turn/Easy_Turn`，github.com/ASLP-lab/Easy-Turn）と公開の重み（`~/2026/work/easy_turn/ckpt/`）が要る．パスは `local/encac/et_model.py` の先頭に書いてある．

```bash
cd egs2/cejc/asr1/local/encac
python train_et.py <asr1>/exp/easyturn_ft_20261101            # 書き起こし＋状態のタグを出すように学習（2 エポック）
bash run_dump.sh                                               # オフラインのスコア（tt_preds/easyturn_ft_*_post5.0.tsv）
```

- 実時間で動かすときは，Easy Turn を別のプロセスのサーバ（exp-realtime-epd の `tools/et_server.py`）で立てて，`rt_eval.py --model easy_turn_ft --et_url http://127.0.0.1:<port>` から呼ぶ．
- 応答の決め方は，実際に流して選ぶ（`local/realtime_eval/kobori_chain23.sh`．サーバを 4 つ立てて 4 並列）．[07 実時間の評価](../07_realtime_eval/README.md) を参照．
- 仕組みを調べたアブレーション：`run_et_notext.sh`（書き起こしを出さない），`run_et_ac.sh`（音響だけ：エンコーダ＋平均＋線形層）．

## Smart Turn

```bash
cd <asr1>        # 元は tt_analysis/smartturn_ft_20261028 に置いて動かした
python local/baselines/smartturn_ft/train_st.py <出力>                       # 公開の ONNX の重みから続けて学習（4 エポック）
SETS=eval_g N=4 bash local/baselines/smartturn_ft/run_dump.sh               # オフラインのスコア
```

完了＝1，継続・相槌＝0 の 2 値で学習する．読み出し位置は区間末の −0.6〜+2.0 秒．

## VAP

```bash
python local/baselines/vap_ft/train_vap.py <出力>                         # Transformer と出力層だけを学習（CPC は固定）
python local/baselines/vap_ft/vap_ft_dump.py data/eval_g tt_preds/vap_ft_eval_g_post5.0.tsv
```

- maai のソース（`~/tt_base/maai_src/maai`）を import して使う．maai は CC BY-NC-ND 4.0 なので，maai のコードと学習し直した重みは配らない．
- ch1＝本人の音声，ch2＝無音で使う．

## オフラインでの比べ方（前の報告まで）

`local/offline_eval/` は，フレームダンプから応答の決め方（2 段の規則）を開発データで選び，評価データで全方式を比べたもの（元は `tt_analysis/longlat_20261024` に置いて動かした）．

```bash
bash local/offline_eval/run_eval_new.sh <名前>     # gt・smartft・vapft など．中の L= を置いた場所に合わせる
```

今の報告は実時間の評価だけを使う．
