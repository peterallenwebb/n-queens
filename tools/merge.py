#!/usr/bin/env python3
"""Merge and verify the phase CSVs written by tools/worker.py (RUNBOOK.md, VALIDATION.md).

    merge.py N --stride S [--dir DIR] [files ...] [--binary PATH] [--json OUT]

Collects every row with this (N, S) from the CSVs, checks that each phase 0..S-1 is present, that
duplicate records of a phase agree (they count as independent verification), sums the phases,
compares the total with OEIS A000170 when N <= 27, and — with --binary — runs the mod-8 (Burnside)
check on the total.  Exit status is 0 only when every phase is present, consistent and (where known)
the total matches.  Missing phases are written to DIR/parity_N_sS_missing.txt for a replacement
worker (`worker.py N --stride S --phases-file ...`).
"""
import argparse, csv, glob, json, os, subprocess, sys

KNOWN = {1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712,
         14: 365596, 15: 2279184, 16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848, 20: 39029188884,
         21: 314666222712, 22: 2691008701644, 23: 24233937684440, 24: 227514171973736, 25: 2207893435808352,
         26: 22317699616364044, 27: 234907967154122528}

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--dir", default="runs")
ap.add_argument("files", nargs="*")
ap.add_argument("--binary", default=None, help="run `binary N --mod8 total` on the merged count")
ap.add_argument("--json", default=None, help="write a summary here (default DIR/parity_N_sS_summary.json)")
a = ap.parse_args()

paths = sorted(set(glob.glob(os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_*.csv")) + a.files))
recs = {}   # phase -> list of (q, placements, wall, source)
hosts = set()
for path in paths:
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["n"]) != a.n or int(row["stride"]) != a.stride: continue
            recs.setdefault(int(row["phase"]), []).append((int(row["q_partial"]), int(row["placements"]), float(row["wall_s"]), os.path.basename(path)))
            hosts.add((row["host"], row["gpu"], row["binary_sha"], row["commit"]))

def ranges(xs):
    xs = sorted(xs); out = []
    for x in xs:
        if out and out[-1][1] == x - 1: out[-1][1] = x
        else: out.append([x, x])
    return ", ".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in out)

missing = [p for p in range(a.stride) if p not in recs]
conflicts = {p: v for p, v in recs.items() if len({q for q, _, _, _ in v}) > 1}
verified = [p for p, v in recs.items() if len(v) > 1 and p not in conflicts]
total = sum(v[0][0] for v in recs.values())
placements = sum(v[0][1] for v in recs.values())
wall = sum(v[0][2] for v in recs.values())
print(f"N={a.n} stride {a.stride}: {len(paths)} CSV files, {len(recs)} of {a.stride} phases, "
      f"{sum(len(v) for v in recs.values())} records, {len(verified)} phases verified by duplicates")
for h in sorted(hosts): print("  worker:", " ".join(h))
if missing: print(f"MISSING {len(missing)} phases: {ranges(missing)}")
for p, v in sorted(conflicts.items()):
    print(f"CONFLICT phase {p}: " + "; ".join(f"{q} ({src})" for q, _, _, src in v))
print(f"total placements {placements} | recorded wall {wall / 3600:.2f} h (sum over phases)")
status = "CONFLICT" if conflicts else "PARTIAL" if missing else "complete"
if missing:
    with open(os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_missing.txt"), "w") as f: f.write("\n".join(map(str, missing)) + "\n")
if not missing and not conflicts:
    if a.n in KNOWN:
        status = "ok (OEIS A000170)" if total == KNOWN[a.n] else f"MISMATCH (OEIS A000170 = {KNOWN[a.n]})"
    else:
        status = "complete (no reference value)"
print(f"Q({a.n}) = {total} {status}" if status != "CONFLICT" else f"Q({a.n}): CONFLICT — resolve the disagreeing phases before trusting any total")
mod8 = None
if a.binary and not missing and not conflicts:
    r = subprocess.run([a.binary, str(a.n), "--mod8", str(total)], capture_output=True, text=True)
    mod8 = (r.stdout + r.stderr).strip(); print(mod8)
out = a.json or os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_summary.json")
with open(out, "w") as f:
    json.dump({"n": a.n, "stride": a.stride, "phases": len(recs), "missing": missing, "conflicts": sorted(conflicts),
               "verified": sorted(verified), "total": total, "placements": placements, "wall_h": wall / 3600,
               "status": status, "mod8": mod8, "workers": sorted(hosts), "files": paths}, f, indent=1)
sys.exit(0 if (not missing and not conflicts and "MISMATCH" not in status and "FAIL" not in (mod8 or "")) else 1)
