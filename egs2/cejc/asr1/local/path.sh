#!/usr/bin/env bash
# path.sh の最後に source される、このレシピ固有の設定。
#
# リポジトリ内の espnet2 / espnet を import できるようにする。
#
# 注意: conda 環境に espnet が **editable install** されている場合、それは PEP 660 の
# meta-path finder として登録されるため PYTHONPATH では上書きできない。その環境では
# `import espnet2` が元の作業ディレクトリを指し続ける。リポジトリ側のコードを使いたい場合は
# リポジトリ直下で `pip install -e .` を実行して editable install を張り替えること
# （詳細は README を参照）。

_ESPNET_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${_ESPNET_REPO_ROOT}:${PYTHONPATH:-}"
unset _ESPNET_REPO_ROOT
