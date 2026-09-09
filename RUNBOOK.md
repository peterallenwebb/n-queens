# Runbook: a fleet run of Q(n)

Everything runs from one operator machine over ssh; nothing but this repository
goes onto a rented GPU box (no credentials of any kind). The tools live in
`tools/`.

## Build and check a box

```
bash tools/ship.sh PORT HOST                   # tar the checkout over ssh, build with --features cuda (refuses a dirty tree)
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

**The work map and the run manifest.** A phase is a set of positions in the
binary's work-item list, so phases are only compatible when they were cut from
the same list. The binary prints a fingerprint of that list ("work map:
fingerprint …", a 64-bit hash of n, the symmetry setting and every item in
order; `nqueens-parity N --items` prints it without running anything). The
first worker of a run writes `runs/parity_N_sS_manifest.json` with the
fingerprint and the item count; every later worker checks its own binary
against the manifest before starting, every phase's output is checked against
it before being recorded, and every record carries it. A worker whose binary
has a different work map refuses to run. Commits that do not touch
`work_items` (documentation, tooling, host code) keep the fingerprint, so a
run may span them; a commit that changes the ordering changes the fingerprint,
and the test `work_map_fingerprints_are_stable` pins the current values so
that such a change is a conscious act.

## Records push

The boxes push their records to the bucket `paw-nqueens-records` (Google
Cloud Storage; uniform access, public access prevention, object versioning),
so nothing is lost when a box dies while nobody is collecting. The only thing
that goes onto a box is a set of **signed PUT URLs**: each allows one thing,
uploading one object, and expires after at most seven days. The
service-account key that signs them (`~/.config/nqueens/gcs-signer.json`,
Storage Object Admin on this bucket only) never leaves the laptop, and the
tools use it directly, so the active `gcloud` account is irrelevant.

```
~/.config/nqueens/venv/bin/python tools/bucket_selftest.py                    # once: bucket settings, a signed PUT, read-back, versioning
~/.config/nqueens/venv/bin/python tools/presign.py 28 --stride 1200 --hosts boxes.txt   # push/<label>.env per box, 7 days
ssh -p PORT HOST 'cat > n-queens/runs/push_urls.env' < push/box01.env        # ship each box its URLs (fleet.py does this)
~/.config/nqueens/venv/bin/python tools/sync.py 28 --stride 1200 --dir runs   # pull the run's records from the bucket
```

The worker uploads the run manifest at start and its CSV and log after every
phase with `curl -T`; a failed upload is logged and retried after the next
phase, and the box's own CSV stays the primary record. Labels come from an
optional third column of `boxes.txt` (`PORT HOST LABEL`), else box01, box02, …
in file order. Signed URLs cannot outlive seven days: for a longer run,
presign again and replace `runs/push_urls.env` on each box; the worker reads
the file at every upload, so no restart is needed. The client library lives in
a private virtualenv: `python3 -m venv ~/.config/nqueens/venv &&
~/.config/nqueens/venv/bin/pip install google-cloud-storage`.

## Collect and merge

```
~/.config/nqueens/venv/bin/python tools/sync.py 28 --stride 1200 --dir runs   # from the bucket (preferred)
bash tools/collect.sh 28 1200 boxes.txt runs                                  # over ssh: boxes.txt lines "PORT HOST"; also pulls the manifest
python3 tools/merge.py 28 --stride 1200 --dir runs --binary ./target/release/nqueens-parity
```

Sync or collect at least hourly; with the push in place the ssh collection is
a fallback and a cross-check of the pushed copies. The merge requires one work-map fingerprint across
every record and the manifest (anything else is INCOMPATIBLE and no total is
printed), every phase 0 … S−1 exactly once (agreeing duplicates count as
verification; a disagreement is a CONFLICT), and phase item counts that add up
to the work map's total; it then sums the phases, compares with OEIS when
n ≤ 27, runs the mod-8 check on the total, writes `parity_N_sS_summary.json`
and, when phases are missing, `parity_N_sS_missing.txt`. Records made before
fingerprints existed are admitted only with `--allow-missing-fingerprint`.
`python3 tools/test_tools.py` exercises all of these paths against a fake
binary.

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
| `NQ_PREF` | prefix rows decided by the ticket; ticket counts use checked arithmetic and a prefix giving more than 2⁴⁰ tickets for one partition is refused | 3 (+1 at n ≥ 24, +1 at n ≥ 28) |
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

- `tools/ship.sh` refuses to ship a dirty working tree, so the commit
  recorded on the box is the commit that was built; it fails loudly on any
  ssh, extraction or build error and prints the binary's SHA-256. Build into
  another target directory (`bash tools/ship.sh PORT HOST target2`) while a
  production run is using `target/release/nqueens-parity`; the worker spawns
  the binary per phase, and swapping it mid-run changes the provenance of
  later phases (the worker refuses a swapped binary with a different work map,
  but not one with the same map).
- Never stop a process with a `pkill -f` pattern that also appears in your
  own ssh command line: the `bash -c` script matches itself. Use the pid file.
- Timing on shared boxes is contaminated by neighbours; counts are not.
