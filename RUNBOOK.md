# Runbook: a fleet run of Q(n)

Everything runs from one operator machine over ssh; nothing but this repository
goes onto a rented GPU box (no credentials of any kind). The tools live in
`tools/`.

## Build and check a box

```
sh tools/ship.sh PORT HOST                     # tar the checkout over ssh, build with --features cuda
ssh -p PORT HOST 'cd n-queens && for n in 16 17 18 19 20 21 22 23; do ./target/release/nqueens-parity $n --cuda | grep -E "parity join|mod-8"; done'
```

Every line must say `ok` and `mod-8 check: PASS`. A fresh Ubuntu image needs
Rust (`curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal`), the
NVIDIA driver, and `libnvrtc` from the CUDA toolkit; `ship.sh` builds with
`~/.cargo/bin/cargo` because a non-login ssh shell has no cargo on its PATH.

## Choose the stride

A phase should cost less than the compute you are willing to lose on an
interruption: S ≈ (expected hours × 60) / target minutes. Each phase holds
about (work items)/S items — keep a few thousand — and costs about a second
of fixed overhead. `nqueens-parity N --items` prints the item count.

| n | work items | one 5090 | S for ~30-minute phases |
|---|---|---|---|
| 26 | 2,602,198 | 6.1 h | 12 |
| 27 | (odd n) | ~88 h | 200 |
| 28 | 10,034,104 | ~500 h (projection) | 1,000 |

## Start the workers

Worker w of W owns the phases ≡ w (mod W). One worker per GPU, any number of
hosts, same (N, S, W) everywhere:

```
ssh -p PORT HOST 'cd n-queens && sh tools/start_worker.sh 28 1000 0 4 --timeout-min 180'   # worker 0 of 4
ssh -p PORT HOST 'cd n-queens && sh tools/start_worker.sh 28 1000 1 4 --gpu 1'           # second GPU of the same box
```

`start_worker.sh` runs `tools/worker.py` detached; the log is
`runs/worker_N_sS_wW.log`, the pid is in `runs/worker_N_sS_wW.pid`, and each
finished phase is fsynced to `runs/parity_N_sS_wWofW_<host>.csv` with its
provenance. Rerunning the same command resumes; a killed worker loses only the
phase in flight; a failing phase is retried and then recorded in `.failed`
while the worker moves on. Stop a worker with `kill $(cat runs/....pid)`.

## Collect and merge

```
sh tools/collect.sh 28 1000 boxes.txt runs         # boxes.txt: one "PORT HOST" per line
python3 tools/merge.py 28 --stride 1000 --dir runs --binary ./target/release/nqueens-parity
```

Collect at least hourly: a box that its provider terminates takes its
uncollected phases with it. The merge sums the phases, requires every phase
0 … S−1 exactly once (agreeing duplicates count as verification; a
disagreement is a CONFLICT and no total is trusted), compares with OEIS when
n ≤ 27, runs the mod-8 check on the total, writes `parity_N_sS_summary.json`
and, when phases are missing, `parity_N_sS_missing.txt`.

## Lost box, verification sample

Hand the missing phases to a new box: `python3 tools/worker.py 28 --stride 1000
--phases-file parity_28_s1000_missing.txt`. For the verification sample
(VALIDATION.md, check 3), rerun chosen phases with the alternate configuration
on any box: `python3 tools/worker.py 28 --stride 1000 --phases 7,311,850 --redo
--tag verify --env NQ_T32=0 --env NQ_RANK2=0`.

## Environment variables of the binary

| variable | meaning | default |
|---|---|---|
| `NQ_PAIR_STRIDE`, `NQ_PAIR_PHASE` | run the work items with index ≡ phase (mod stride) | 1, 0 |
| `NQ_PAIR_FROM`, `NQ_PAIR_TO` | restrict to the index range [from, to) | whole list |
| `NQ_PARITY_MEM` | device memory for the batch tables, bytes | 8 GiB |
| `NQ_PREF` | prefix rows decided by the ticket | 3 (+1 at n ≥ 24, +1 at n ≥ 28) |
| `NQ_THREADS_GPU` | persistent threads | 139,264 |
| `NQ_MAXSTACK` | frames per lane in the shared-memory stack (checked against every item) | n − prefix + 1 |
| `NQ_SLM` | block size | 128 when the static shared memory fits, else 64 |
| `NQ_RANK2` | two-level shared-memory ranks (0 = flat table) | 1 |
| `NQ_T32` | u32 table entries (0 = u64) | 1 |
| `NQ_TABMAX` | report the largest table entry of the run | 0 |
| `NQ_MAXBATCH` | stop after this many batches (a probe; partial count) | unlimited |
| `NQ_VERBOSE` | per-batch progress lines | 0 |
| `NQ_NVRTC_ARCH` | NVRTC target override (e.g. sm_120) | the device's |

The binary prints `parity join: Q=<count> <status> | placements … | … | <wall>s`
with status `ok`, `MISMATCH` (against OEIS), `unknown` (no reference) or
`PARTIAL` (a selection), then the mod-8 line for full runs; exit status 0 on
`ok`/`unknown` with mod-8 PASS, 3 on a mismatch or a failed check.

## Operating notes

- Build into another target directory (`sh tools/ship.sh PORT HOST target2`)
  while a production run is using `target/release/nqueens-parity`; the worker
  spawns the binary per phase, and swapping it mid-run changes the provenance
  of later phases.
- Never stop a process with a `pkill -f` pattern that also appears in your
  own ssh command line: the `bash -c` script matches itself. Use the pid file.
- Timing on shared boxes is contaminated by neighbours; counts are not.
