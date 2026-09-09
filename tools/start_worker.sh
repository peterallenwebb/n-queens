#!/bin/sh
# Start one worker on a GPU box (run on the box, in the n-queens checkout):
#     sh tools/start_worker.sh N STRIDE WORKER WORKERS [extra worker.py args...]
# The worker runs detached under nohup; progress log runs/worker_N_sS_wW.log, pid file
# runs/worker_N_sS_wW.pid, results runs/parity_N_sS_wWofW_<host>.csv.  Rerunning the same command
# resumes.  Stop it with `kill $(cat runs/worker_N_sS_wW.pid)` — never with a pkill pattern that also
# appears in your own ssh command line.
set -e
N=$1; S=$2; W=$3; NW=$4; shift 4
mkdir -p runs
LOG=runs/worker_${N}_s${S}_w${W}.log; PID=runs/worker_${N}_s${S}_w${W}.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "worker already running, pid $(cat "$PID") (see $LOG)"; exit 1; fi
(nohup python3 tools/worker.py "$N" --stride "$S" --worker "$W" --workers "$NW" "$@" >> "$LOG" 2>&1 < /dev/null & echo $! > "$PID")
sleep 1; echo "started pid $(cat "$PID"), log $LOG"
