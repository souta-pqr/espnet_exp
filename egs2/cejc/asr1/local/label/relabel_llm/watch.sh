#!/bin/bash
# cron から 5 分おきに呼ばれる。GPU に他ユーザーのプロセスが 1 つも無い状態が
# QUIET_MIN 分続いたら、guard.py の下で run_full_main.sh を切り離して起動する。
# guard.py は他ユーザーの GPU プロセスが現れたら本番を止める（rc=2）。その後は次の空きを待って再開する。
# 全件終わった（done）・確認で止まった（gate_failed）・想定外のエラーのときは README を書き、自分の cron エントリを消す。
set -u
cd ~/relabel_llm
QUIET_MIN=15
log() { echo "$(date '+%F %T') $*" >> logs/watch.log; }
finish() {
  log "終了処理: $1"
  ~/anaconda3/envs/llm/bin/python finalize_readme.py "$1" >> logs/watch.log 2>&1
  crontab -l 2>/dev/null | grep -v 'relabel_llm/watch.sh' | crontab -
  exit 0
}

flock -n logs/run.lock true || exit 0                  # 本番が動いている

state=$(cat logs/state 2>/dev/null || true)
rc=$(cat logs/last_rc 2>/dev/null || true)
[ "$state" = done ] && finish done
[ "$state" = gate_failed ] && finish gate_failed
[ "$state" = error ] && finish error
case "$rc" in
  ""|0|2|3) ;;                                         # 未起動・他ユーザーで停止・他ユーザーがいて起動せず
  *) finish "error(rc=$rc)" ;;
esac

foreign=$(/usr/bin/python3 -c 'import guard; print(" ".join(guard.foreign_gpu_users()))')
if [ -n "$foreign" ]; then
  touch logs/foreign_seen
  exit 0
fi
if [ -e logs/foreign_seen ] && [ -n "$(find logs/foreign_seen -mmin -$QUIET_MIN)" ]; then
  exit 0                                               # 空いてから QUIET_MIN 分たっていない
fi

log "GPU が空いている（他ユーザーのプロセスなし、${QUIET_MIN} 分以上）→ 起動"
rm -f logs/last_rc
setsid nohup flock -n logs/run.lock bash -c \
  '/usr/bin/python3 guard.py logs/full_main.log -- ./run_full_main.sh >> logs/full_main.guard.log 2>&1; echo $? > logs/last_rc' \
  > /dev/null 2>&1 < /dev/null &
