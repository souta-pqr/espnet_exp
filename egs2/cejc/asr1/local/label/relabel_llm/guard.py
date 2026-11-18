#!/usr/bin/env python3
"""GPU ジョブを起動し、自分（kobori）以外のユーザーが GPU を使い始めたら即座に止める。

silver9 では mogi さんの GPU 予約が優先なので、他ユーザーの GPU プロセスが見えたら
ジョブのプロセスグループごと SIGTERM → 10 秒後に SIGKILL する。起動時に既に他ユーザーがいれば起動しない。
ジョブは途中再開できる前提（run_gold.py は済んだ分を飛ばす）。

  setsid nohup python3 guard.py logs/xxx.log -- <command ...> > logs/xxx.guard.log 2>&1 < /dev/null &
"""
import getpass
import os
import signal
import subprocess
import sys
import time

ME = getpass.getuser()
INTERVAL = 3


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def foreign_gpu_users():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    users = []
    for p in out.split():
        if not p.strip().isdigit():
            continue
        u = subprocess.run(["ps", "-o", "user=", "-p", p], capture_output=True, text=True).stdout.strip()
        if u != ME:                       # 持ち主が分からないものも他人扱い（安全側）
            users.append(f"{u or '?'}(pid {p})")
    return users


def main():
    i = sys.argv.index("--")
    joblog, cmd = sys.argv[1], sys.argv[i + 1:]
    f = foreign_gpu_users()
    if f:
        log(f"他ユーザーが GPU 使用中のため起動しない: {f}")
        return 3
    with open(joblog, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    log(f"起動 pid={proc.pid}: {' '.join(cmd)}")
    while True:
        rc = proc.poll()
        if rc is not None:
            log(f"ジョブ終了 rc={rc}")
            return rc
        f = foreign_gpu_users()
        if f:
            log(f"他ユーザーの GPU 使用を検出 → ジョブ停止: {f}")
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(1)
                if proc.poll() is not None:
                    break
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            log("停止完了")
            return 2
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
