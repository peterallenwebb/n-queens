#!/usr/bin/env python3
"""Merge and verify the phase CSVs written by tools/worker.py (RUNBOOK.md, VALIDATION.md).

    merge.py N --stride S [--dir DIR] [files ...] [--binary PATH] [--json OUT] [--allow-missing-fingerprint]

Collects every row with this (N, S) from the CSVs and checks, in this order: that every record carries
the same work-map fingerprint (and the run manifest's, when DIR/parity_N_sS_manifest.json exists) — phases
are positions in the binary's work-item list, so records of different work maps must never be added;
that each phase 0..S-1 is present; that duplicate records of a phase agree (they count as independent
verification); that the recorded item counts of the phases add up to the work map's total.  It then sums
the phases, compares the total with OEIS A000170 when N <= 27, and — with --binary — runs the mod-8
(Burnside) check on the total.  Exit status is 0 only when everything above holds.  Missing phases are
written to DIR/parity_N_sS_missing.txt for a replacement worker (`worker.py N --stride S --phases-file ...`).
Records made by the pre-fingerprint tool have no fingerprint; --allow-missing-fingerprint admits them.
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
ap.add_argument("--allow-missing-fingerprint", action="store_true", help="admit records made before fingerprints existed")
a = ap.parse_args()

paths = sorted(set(glob.glob(os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_*.csv")) + a.files))
manifest_path = os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_manifest.json")
manifest = None
if os.path.exists(manifest_path):
    with open(manifest_path) as f: manifest = json.load(f)
recs = {}          # phase -> list of (q, placements, items, wall, source)
hosts = set()
fps = {}           # fingerprint (or None) -> set of source files
totals = set()     # recorded items_total values
for path in paths:
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["n"]) != a.n or int(row["stride"]) != a.stride: continue
            fp = row.get("workmap") or None
            fps.setdefault(fp, set()).add(os.path.basename(path))
            if row.get("items_total"): totals.add(int(row["items_total"]))
            recs.setdefault(int(row["phase"]), []).append((int(row["q_partial"]), int(row["placements"]), int(row.get("items") or -1), float(row["wall_s"]), os.path.basename(path)))
            hosts.add((row["host"], row["gpu"], row["binary_sha"], row["commit"]))

def ranges(xs):
    xs = sorted(xs); out = []
    for x in xs:
        if out and out[-1][1] == x - 1: out[-1][1] = x
        else: out.append([x, x])
    return ", ".join(f"{lo}" if lo == hi else f"{lo}-{hi}" for lo, hi in out)

problems = []
known_fps = {fp for fp in fps if fp is not None}
if manifest is not None and manifest.get("workmap"): known_fps_with_manifest = known_fps | {manifest["workmap"]}
else: known_fps_with_manifest = known_fps
if len(known_fps_with_manifest) > 1:
    problems.append("INCOMPATIBLE work maps: " + "; ".join(f"{fp} in {sorted(srcs)}" for fp, srcs in fps.items() if fp)
                    + (f"; manifest {manifest['workmap']}" if manifest and manifest.get("workmap") not in known_fps else ""))
if None in fps:
    msg = f"{len(fps[None])} file(s) hold records without a work-map fingerprint: {sorted(fps[None])}"
    if a.allow_missing_fingerprint: print("WARNING (admitted): " + msg)
    else: problems.append("INCOMPATIBLE: " + msg + " (pass --allow-missing-fingerprint for pre-fingerprint records)")
if manifest is not None:
    for key, val in (("n", a.n), ("stride", a.stride)):
        if manifest.get(key) != val: problems.append(f"manifest {key} is {manifest.get(key)}, expected {val}")
    if manifest.get("items_total"): totals.add(int(manifest["items_total"]))
if len(totals) > 1: problems.append(f"INCOMPATIBLE: records disagree on the number of work items: {sorted(totals)}")

missing = [p for p in range(a.stride) if p not in recs]
conflicts = {p: v for p, v in recs.items() if len({q for q, _, _, _, _ in v}) > 1}
verified = [p for p, v in recs.items() if len(v) > 1 and p not in conflicts]
total = sum(v[0][0] for v in recs.values())
placements = sum(v[0][1] for v in recs.values())
wall = sum(v[0][3] for v in recs.values())
items_sum = sum(v[0][2] for v in recs.values()) if recs and all(v[0][2] >= 0 for v in recs.values()) else None
items_total = next(iter(totals)) if len(totals) == 1 else None
print(f"N={a.n} stride {a.stride}: {len(paths)} CSV files, {len(recs)} of {a.stride} phases, "
      f"{sum(len(v) for v in recs.values())} records, {len(verified)} phases verified by duplicates"
      + (f" | work map {next(iter(known_fps))}" if len(known_fps) == 1 else ""))
for h in sorted(hosts): print("  worker:", " ".join(h))
for pr in problems: print(pr)
if missing: print(f"MISSING {len(missing)} phases: {ranges(missing)}")
for p, v in sorted(conflicts.items()):
    print(f"CONFLICT phase {p}: " + "; ".join(f"{q} ({src})" for q, _, _, _, src in v))
if not missing and items_sum is not None and items_total is not None and items_sum != items_total:
    problems.append(f"ITEMS MISMATCH: the phases record {items_sum} work items, the work map has {items_total}")
    print(problems[-1])
print(f"total placements {placements} | recorded wall {wall / 3600:.2f} h (sum over phases)"
      + (f" | items {items_sum} of {items_total}" if items_sum is not None and items_total is not None else ""))
if conflicts: status = "CONFLICT"
elif problems: status = "INCOMPATIBLE"
elif missing: status = "PARTIAL"
elif a.n in KNOWN: status = "ok (OEIS A000170)" if total == KNOWN[a.n] else f"MISMATCH (OEIS A000170 = {KNOWN[a.n]})"
else: status = "complete (no reference value)"
if missing:
    with open(os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_missing.txt"), "w") as f: f.write("\n".join(map(str, missing)) + "\n")
if status in ("CONFLICT", "INCOMPATIBLE"): print(f"Q({a.n}): {status} — no total can be trusted until this is resolved")
else: print(f"Q({a.n}) = {total} {status}")
mod8 = None
if a.binary and status not in ("CONFLICT", "INCOMPATIBLE", "PARTIAL"):
    r = subprocess.run([a.binary, str(a.n), "--mod8", str(total)], capture_output=True, text=True)
    mod8 = (r.stdout + r.stderr).strip(); print(mod8)
out = a.json or os.path.join(a.dir, f"parity_{a.n}_s{a.stride}_summary.json")
with open(out, "w") as f:
    json.dump({"n": a.n, "stride": a.stride, "phases": len(recs), "missing": missing, "conflicts": sorted(conflicts),
               "verified": sorted(verified), "total": total, "placements": placements, "wall_h": wall / 3600,
               "items_sum": items_sum, "items_total": items_total, "workmap": next(iter(known_fps)) if len(known_fps) == 1 else sorted(known_fps),
               "manifest": manifest, "status": status, "mod8": mod8, "workers": sorted(hosts), "files": paths, "problems": problems}, f, indent=1)
ok = status.startswith("ok") or status.startswith("complete")
sys.exit(0 if ok and "FAIL" not in (mod8 or "") else 1)
