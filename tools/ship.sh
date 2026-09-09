#!/bin/sh
# Ship this checkout to a rented GPU box and build there (nothing but the repository leaves this
# machine — no credentials of any kind go onto a rented box):
#     sh tools/ship.sh PORT HOST [TARGET_DIR]        e.g.  sh tools/ship.sh 30143 root@ssh1.vast.ai
# Sends tracked + untracked (non-ignored) files as a tar stream over ssh, without macOS xattrs, plus
# a COMMIT file for provenance; builds with cargo from ~/.cargo/bin (a non-login ssh shell has no
# cargo on PATH) into TARGET_DIR (default target).  Build into another directory while a production
# run is using target/release/nqueens-parity: the worker spawns the binary per phase, and replacing
# it mid-run would change the provenance of later phases.  The ssh key path is ~/.ssh/vast_nqueens.
set -e
PORT=$1; HOST=$2; TD=${3:-target}
git rev-parse --short HEAD > COMMIT
{ git ls-files -z --cached --others --exclude-standard; printf 'COMMIT\0'; } | COPYFILE_DISABLE=1 tar --null --no-xattrs -T - -czf - 2>/dev/null \
  | ssh -i ~/.ssh/vast_nqueens -p "$PORT" -o ConnectTimeout=20 "$HOST" \
      "mkdir -p n-queens && cd n-queens && tar -xzf - 2>&1 | grep -v 'extended header'; export PATH=\$HOME/.cargo/bin:\$PATH; \
       CARGO_TARGET_DIR=$TD cargo build --release --features cuda > build_$TD.log 2>&1; grep -E '^error|command not found' -A5 build_$TD.log | head -20; \
       tail -n 1 build_$TD.log; ls -l $TD/release/nqueens-parity" 2>&1 | grep -v '^Welcome\|^Have fun\|^AI agents'
