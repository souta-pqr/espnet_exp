# VAP-lite turn-taking head

凍結ストリーミング ASR エンコーダの上に載せる、frame-level の発話区間末/相槌判定ブランチ。
**ASR とエンコーダは一切変更しない**（凍結）。turn-taking は ASR と並走する。

## 構成
- `model.py` — `TurnTakingHead`: 因果 dilated conv(過去のみ) + 2ヘッド
  - `cls`: 継続/区間末/相槌 を毎フレーム (3class)
  - `va`: 対象話者が今後 h 秒以内に発話するか (自己教師, free label)
- `extract_features.py` — 凍結 encoder_out (T,256) と未来VAラベルを抽出しシャード保存
- `train.py` — 時間重み付き分類CE + λ·VA-BCE でヘッドのみ学習
- `evaluate.py` — 停止規則(TEASER 閾値連続 / SPRT 累積)で「精度 vs 早期性」を計測
- `run.sh` — 抽出→学習→評価 を一括実行

## 設計の肝
- frame-level（CIF を使わない）: CIF は最終語で発火が止まり**沈黙で動かない**。区間末の
  決定打は区間末以降の無音/次ターンにあるので、frame 単位で連続出力する必要がある。
- 因果のみ: 左パディング conv = 未来を見ない。ストリーミング整合。
- 適応遅延: 明確な発話は早期確定、曖昧な発話は無音を待つ（SPRT/ECONOMY）。ASR の遅延は増えない。
- 精度天井 ~0.70（frozen encoder の no/yes 分離限界, methodA_teaser §9）。本ブランチの利得は
  **早期性・適応性・自己教師信号**であって精度天井の引き上げではない。

## 実行
```bash
# スモークテスト(各200件)
MAX_UTTS=200 bash local/turntaking/run.sh 1     # 全stage
# 本番
bash local/turntaking/run.sh 1                  # stage1(抽出)から全部
bash local/turntaking/run.sh 2 2                # 学習のみ
bash local/turntaking/run.sh 3 3                # 評価のみ
```
conda env は `/home/kobori/.conda/envs/espnet` 固定。
