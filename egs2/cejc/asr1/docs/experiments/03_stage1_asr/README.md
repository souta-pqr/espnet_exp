# 03 音声認識（Stage1）

CEJC で音声認識（Transducer，エンコーダは contextual block conformer）を学習する．
このエンコーダを，04 のエンコーダ固定＋Transformer と 05 の提案手法がそのまま使う．

[← 全体の README](../../../../../../README.md)

## 設定

- 設定：`myconf/train_asr_pureasr_conformer_blk42_sp_g.yaml`（block_size 42・速度摂動 0.9/1.0/1.1）
- タグ：`20261006-pureasr-g-blk42-sp` → 重みは `exp/asr_20261006-pureasr-g-blk42-sp/valid.loss.ave.pth`
- エンコーダ：12 層・出力 256 次元．先読みは 100 ms．

## 手順

[01 データの準備](../01_data/README.md) の 2 まで済ませてから流す．

```bash
setsid nohup bash tt_batch_logs/run_g_stage1.sh >> tt_batch_logs/g_stage1.log 2>&1 < /dev/null &
```

段ごとに `tt_batch_logs/g_stage1_<段>.done` を置く．落ちたら直して流し直せば，済んだ段は飛ばす．

1. 学習（GPU のメモリが足りなければ batch_bins を半分にして最大 2 回やり直す）
2. `dump/raw/train_g`（速度摂動の 1.0 倍だけ）を作る
3. `dev_g`・`eval_g`・`train_g` をデコードする（CER は `exp/asr_<タグ>/RESULTS.md`）

## 注意

- `block_size` を変えると，エンコーダの出力が変わるので，04・05 は全部やり直しになる．
- 以前の改善の記録は `docs/stage1_asr_improvement.md`．
