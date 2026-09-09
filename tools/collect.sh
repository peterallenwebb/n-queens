#!/bin/bash
# Pull the phase CSVs and the run manifest of one run from every box listed in a hosts file:
#     bash tools/collect.sh N STRIDE boxes.txt [DIR]
# boxes.txt has one box per line: "PORT HOST" (e.g. "30143 root@ssh1.vast.ai").  Files stream through ssh
# as a tar; a box that cannot be reached or has nothing to collect is reported and the exit status is 1.
# Then:  python3 tools/merge.py N --stride STRIDE --dir DIR --binary ./target/release/nqueens-parity
# Collect at least hourly during a run: a box that its provider terminates takes its uncollected phases with it.
set -euo pipefail
N=${1:?N}; S=${2:?STRIDE}; BOXES=${3:?boxes file}; DIR=${4:-runs}
mkdir -p "$DIR"; fail=0
while read -r PORT HOST; do
  [ -z "${PORT:-}" ] && continue
  echo "== $HOST:$PORT"
  if ! ssh -i ~/.ssh/vast_nqueens -p "$PORT" -o ConnectTimeout=20 "$HOST" \
        "cd n-queens/runs && tar -czf - \$(ls parity_${N}_s${S}_*.csv parity_${N}_s${S}_manifest.json 2>/dev/null)" 2>/dev/null \
        | tar -xzf - -C "$DIR"; then
    echo "collect.sh: FAILED for $HOST:$PORT"; fail=1
  fi
done < "$BOXES"
ls -l "$DIR"/parity_${N}_s${S}_*.csv 2>/dev/null || true
exit $fail
