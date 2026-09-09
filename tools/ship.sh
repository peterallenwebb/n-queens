#!/bin/bash
# Ship this checkout to a rented GPU box and build there (nothing but the repository leaves this
# machine — no credentials of any kind go onto a rented box):
#     bash tools/ship.sh PORT HOST [TARGET_DIR]        e.g.  bash tools/ship.sh 30143 root@ssh1.vast.ai
# Refuses a dirty working tree (ALLOW_DIRTY=1 overrides; the commit is then recorded as <hash>-dirty), so
# the COMMIT file that travels with the tree is truthful.  Sends tracked + untracked (non-ignored) files as
# a tar stream over ssh without macOS xattrs; builds with ~/.cargo/bin/cargo (a non-login ssh shell has no
# cargo on PATH) into TARGET_DIR (default target); fails loudly on any ssh, extraction or build error and
# prints the built binary's SHA-256.  Build into another directory while a production run is using
# target/release/nqueens-parity: the worker spawns the binary per phase, and replacing it mid-run would
# change the provenance of later phases.  The ssh key path is ~/.ssh/vast_nqueens.
set -euo pipefail
PORT=${1:?PORT}; HOST=${2:?HOST}; TD=${3:-target}
if [ -n "$(git status --porcelain)" ]; then
  if [ "${ALLOW_DIRTY:-0}" != 1 ]; then echo "ship.sh: working tree is dirty — commit first (ALLOW_DIRTY=1 to ship anyway)"; git status --short | head -n 20; exit 1; fi
  echo "$(git rev-parse --short HEAD)-dirty" > COMMIT
else
  git rev-parse --short HEAD > COMMIT
fi
remote="set -e; mkdir -p n-queens && cd n-queens; tar -xzf -; export PATH=\$HOME/.cargo/bin:\$PATH; \
  CARGO_TARGET_DIR=$TD cargo build --release --features cuda > build_$TD.log 2>&1 || { echo 'BUILD FAILED'; tail -n 40 build_$TD.log; exit 1; }; \
  test -x $TD/release/nqueens-parity; echo \"built \$(cat COMMIT) -> $TD/release/nqueens-parity sha256 \$(sha256sum $TD/release/nqueens-parity | cut -c1-16)\""
out=$(mktemp)
if { git ls-files -z --cached --others --exclude-standard; printf 'COMMIT\0'; } \
     | COPYFILE_DISABLE=1 tar --null --no-xattrs -T - -czf - 2>/dev/null \
     | ssh -i ~/.ssh/vast_nqueens -p "$PORT" -o ConnectTimeout=20 "$HOST" "$remote" > "$out" 2>&1; then
  grep -v '^Welcome to vast\|^Have fun\|^AI agents' "$out" || true
  rm -f "$out"
else
  rc=$?
  grep -v '^Welcome to vast\|^Have fun\|^AI agents' "$out" || true
  rm -f "$out"
  echo "ship.sh: FAILED (exit $rc)"; exit 1
fi
