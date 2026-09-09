#!/bin/sh
# Pull the phase CSVs of one run from every box listed in a hosts file into a local directory:
#     sh tools/collect.sh N STRIDE boxes.txt [DIR]
# boxes.txt has one box per line: "PORT HOST" (e.g. "30143 root@ssh1.vast.ai").  Files stream
# through ssh as a tar.  Then:  python3 tools/merge.py N --stride STRIDE --dir DIR --binary ./target/release/nqueens-parity
# Collect at least hourly during a run: a box that its host terminates takes its uncollected phases with it.
set -e
N=$1; S=$2; BOXES=$3; DIR=${4:-runs}
mkdir -p "$DIR"
while read -r PORT HOST; do
  [ -z "$PORT" ] && continue
  echo "== $HOST:$PORT"
  ssh -i ~/.ssh/vast_nqueens -p "$PORT" -o ConnectTimeout=20 "$HOST" \
      "cd n-queens/runs && tar -czf - parity_${N}_s${S}_*.csv" 2>/dev/null | tar -xzf - -C "$DIR"
done < "$BOXES"
ls -l "$DIR"/parity_${N}_s${S}_*.csv
