#!/usr/bin/env python3
"""Strided, restartable, multi-worker driver for a fleet run of `nqueens-parity` (RUNBOOK.md).

The binary's work-item list (heaviest first) is split into STRIDE phases: phase p is the work items
with index ≡ p (mod STRIDE).  Every phase samples heavy and light items alike, so phases take about
the same time, and the phases sum to Q(N).  Worker w of W runs the phases p ≡ w (mod W): one worker
per GPU, on any number of hosts, with no coordination beyond the shared (N, STRIDE, W).  Every
finished phase is appended and fsynced to a CSV with its provenance (host, GPU, binary hash, commit,
NQ_* environment); a killed worker loses at most the phase in flight, and rerunning the same command
resumes.  tools/merge.py sums the CSVs of all workers and checks that every phase is present exactly
once (duplicates must agree; they count as verification).

    worker.py N --stride S [--worker w --workers W | --phases a,b,c | --phases lo:hi | --phases-file F]
              [--gpu i] [--binary PATH] [--out DIR] [--env K=V ...] [--tag T]
              [--retries R] [--timeout-min M] [--target-min T] [--redo] [--dry]
"""
import argparse, csv, glob, hashlib, json, os, re, socket, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--phases", default=None, help="explicit phases: a,b,c or lo:hi (overrides --worker/--workers)")
ap.add_argument("--phases-file", default=None, help="explicit phases, one per line (e.g. the merge tool's *_missing.txt)")
ap.add_argument("--gpu", type=int, default=None, help="pin to this GPU (CUDA_VISIBLE_DEVICES)")
ap.add_argument("--binary", default="./target/release/nqueens-parity")
ap.add_argument("--out", default="runs")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--tag", default="", help="label added to the CSV name (e.g. verify)")
ap.add_argument("--retries", type=int, default=2)
ap.add_argument("--timeout-min", type=float, default=0.0, help="kill a phase after this long (0 = never)")
ap.add_argument("--target-min", type=float, default=30.0, help="phase length to aim for (used for the stride hint)")
ap.add_argument("--redo", action="store_true", help="run the phases even if already recorded (verification reruns)")
ap.add_argument("--dry", action="store_true", help="print the plan and exit")
a = ap.parse_args()
if a.stride < 1 or a.workers < 1 or not (0 <= a.worker < a.workers): sys.exit("bad --stride/--worker/--workers")

env = dict(os.environ)
for kv in a.env:
    k, v = kv.split("=", 1); env[k] = v
if a.gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
for k in ("NQ_PAIR_STRIDE", "NQ_PAIR_PHASE", "NQ_PAIR_FROM", "NQ_PAIR_TO", "NQ_MAXBATCH"):
    env.pop(k, None)   # the driver owns the chunking
nq_env = " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("NQ_") or k == "CUDA_VISIBLE_DEVICES")

if a.phases_file:
    with open(a.phases_file) as f: phases = [int(x) for x in f.read().split()]
elif a.phases:
    if ":" in a.phases:
        lo, hi = a.phases.split(":"); phases = list(range(int(lo), int(hi)))
    else:
        phases = [int(x) for x in a.phases.split(",")]
else:
    phases = list(range(a.worker, a.stride, a.workers))
if any(p < 0 or p >= a.stride for p in phases): sys.exit("phase out of range")

host = socket.gethostname()
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception: return ""
gpu_name = sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1") or "unknown"
commit = sh("git rev-parse --short HEAD 2>/dev/null || cat COMMIT 2>/dev/null") or "unknown"   # COMMIT: written by tools/ship.sh
with open(a.binary, "rb") as f: binary_sha = hashlib.sha256(f.read()).hexdigest()[:16]

os.makedirs(a.out, exist_ok=True)
tag = f"_{a.tag}" if a.tag else ""
base = os.path.join(a.out, f"parity_{a.n}_s{a.stride}_w{a.worker}of{a.workers}_{host}{tag}")
csvpath, logpath, failpath = base + ".csv", base + ".log", base + ".failed"
FIELDS = ["n", "stride", "phase", "q_partial", "placements", "items", "wall_s", "finished_at",
          "host", "gpu", "binary_sha", "commit", "env"]

# phases already recorded by any CSV of this (n, stride) in the output directory
done = {}
for path in glob.glob(os.path.join(a.out, f"parity_{a.n}_s{a.stride}_*.csv")):
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["n"]) == a.n and int(row["stride"]) == a.stride:
                done[int(row["phase"])] = int(row["q_partial"])
todo = phases if a.redo else [p for p in phases if p not in done]
print(f"N={a.n} stride {a.stride}: worker {a.worker}/{a.workers} owns {len(phases)} phases, {len(todo)} to do "
      f"({len(done)} of {a.stride} recorded in {a.out}) | {host} {gpu_name} | binary {binary_sha} commit {commit}", flush=True)
if nq_env: print(f"env: {nq_env}", flush=True)
if a.dry: print("phases:", todo[:20], "..." if len(todo) > 20 else ""); sys.exit(0)

def run_phase(p):
    e = dict(env); e["NQ_PAIR_STRIDE"] = str(a.stride); e["NQ_PAIR_PHASE"] = str(p)
    timeout = a.timeout_min * 60 if a.timeout_min > 0 else None
    try:
        r = subprocess.run([a.binary, str(a.n), "--cuda"], env=e, capture_output=True, text=True, timeout=timeout)
        out, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as ex:
        out, code = (ex.stdout or b"").decode(errors="replace") + "\n[timeout]", -1
    with open(logpath, "a") as lf:
        lf.write(f"===== phase {p} exit {code} {time.strftime('%Y-%m-%d %H:%M:%S')}\n{out}\n")
    m = re.search(r"parity join: Q=(\d+) (\w+) \| placements (\d+)", out)
    if code != 0 or not m or m.group(2) not in ("PARTIAL", "ok"): return None
    items = re.search(r"stride \d+ phase \d+: (\d+) items", out)
    return int(m.group(1)), int(m.group(3)), int(items.group(1)) if items else -1

new = not os.path.exists(csvpath)
failed = []
walls = []
with open(csvpath, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(FIELDS); f.flush(); os.fsync(f.fileno())
    for i, p in enumerate(todo):
        res = None
        for attempt in range(a.retries + 1):
            t0 = time.time(); res = run_phase(p); wall = time.time() - t0
            if res is not None: break
            print(f"phase {p}: attempt {attempt + 1} failed after {wall:.0f}s (see {logpath})", flush=True)
            time.sleep(30)
        if res is None:
            failed.append(p)
            with open(failpath, "a") as ff: ff.write(f"{p} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            continue
        q, pl, items = res
        w.writerow([a.n, a.stride, p, q, pl, items, f"{wall:.1f}", time.strftime("%Y-%m-%d %H:%M:%S"),
                    host, gpu_name, binary_sha, commit, nq_env])
        f.flush(); os.fsync(f.fileno())
        done[p] = q; walls.append(wall)
        avg = sum(walls) / len(walls); left = len(todo) - i - 1
        hint = max(1, round(a.stride * avg / (a.target_min * 60)))
        print(f"phase {p}: Q_partial={q} placements={pl} items={items} wall={wall:.0f}s | {i + 1}/{len(todo)} done, "
              f"avg {avg:.0f}s, ETA {left * avg / 3600:.2f} h | stride for {a.target_min:.0f}-min phases ≈ {hint}", flush=True)
print(f"worker {a.worker}/{a.workers}: {len(todo) - len(failed)} phases run, {len(failed)} failed {failed if failed else ''}; "
      f"{len(done)} of {a.stride} phases recorded in {a.out}; merge with tools/merge.py {a.n} --stride {a.stride} --dir {a.out}")
sys.exit(1 if failed else 0)
