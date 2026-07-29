#!/usr/bin/env bash
# egs2/*/*/path.sh から source される Python 環境の有効化スクリプト。
#
# 本家 ESPnet では tools/setup_python.sh などが生成するマシン固有ファイルで、
# 通常はリポジトリに含めない。ここでは run_*.sh を叩くだけで動くようにするため
# 明示的に置いている。**別マシンで使う場合はここを書き換えること。**
#
# これが無いと path.sh は警告を出して素の python3 を使い、
#   ModuleNotFoundError: No module named 'espnet2'
# で全ジョブが即死する。

ESPNET_CONDA_BASE="${ESPNET_CONDA_BASE:-/opt/anaconda3}"
ESPNET_CONDA_ENV="${ESPNET_CONDA_ENV:-/home/kobori/.conda/envs/espnet}"

if [ -f "${ESPNET_CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "${ESPNET_CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ESPNET_CONDA_ENV}" 2>/dev/null \
        || export PATH="${ESPNET_CONDA_ENV}/bin:${PATH}"
else
    export PATH="${ESPNET_CONDA_ENV}/bin:${PATH}"
fi
