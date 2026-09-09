#!/usr/bin/env python3
"""Tests of tools/worker.py and tools/merge.py against a fake binary (no GPU needed): python3 tools/test_tools.py"""
import csv, json, os, subprocess, sys, tempfile, textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER, MERGE = os.path.join(HERE, "worker.py"), os.path.join(HERE, "merge.py")
FP_A, FP_B = "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"

def fake_binary(path):
    """Answers like nqueens-parity: `N --items`, `N --cuda` (a phase of NQ_PAIR_STRIDE/PHASE), `N --mod8 X`."""
    with open(path, "w") as f:
        f.write(textwrap.dedent("""\
            #!/usr/bin/env python3
            import os, sys
            n = int(sys.argv[1]); fp = os.environ.get("FAKE_FP", "%s")
            if "--items" in sys.argv:
                print(f"N={n}: 10 work items (sym 180°), fingerprint {fp}"); sys.exit(0)
            if "--mod8" in sys.argv:
                print("mod-8 check: PASS (fake)"); sys.exit(0)
            s, p = int(os.environ["NQ_PAIR_STRIDE"]), int(os.environ["NQ_PAIR_PHASE"])
            items = len([i for i in range(10) if i %% s == p])
            print(f"chunk: work items [0, 10) of 10, stride {s} phase {p}: {items} items — partial count")
            print(f"work map: fingerprint {fp}, 10 items")
            q = 46 if s == 2 else 92 // s
            print(f"parity join: Q={q} PARTIAL | placements {q * 10} (1.0/sol, 1e9/s) | {items} pairs | 2 launches | 1 batches | 0.1s")
            """) % FP_A)
    os.chmod(path, 0o755)

def run(args, env=None, cwd=None):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run([sys.executable] + args, env=e, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

def write_csv(path, rows, fields=None):
    fields = fields or ["n", "stride", "phase", "q_partial", "placements", "items", "wall_s", "finished_at", "host", "gpu", "binary_sha", "commit", "env", "workmap", "items_total"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})

def row(phase, q, fp=FP_A, items=5, src="h1", total=10):
    return {"n": 8, "stride": 2, "phase": phase, "q_partial": q, "placements": q * 10, "items": items, "wall_s": 1.0, "finished_at": "t",
            "host": src, "gpu": "g", "binary_sha": "b", "commit": "c", "env": "", "workmap": fp, "items_total": total}

fails = 0
def check(name, cond, detail=""):
    global fails
    print(("ok   " if cond else "FAIL ") + name + ("" if cond else "\n" + detail))
    if not cond: fails += 1

with tempfile.TemporaryDirectory() as tmp:
    binary = os.path.join(tmp, "fake-nqueens-parity"); fake_binary(binary)
    out = os.path.join(tmp, "runs")
    # 1. a complete run through the worker: manifest written, phases recorded with the fingerprint, merge ok
    rc, o = run([WORKER, "8", "--stride", "2", "--binary", binary, "--out", out, "--retries", "0"])
    check("worker completes a 2-phase run", rc == 0 and "2 phases run" in o, o)
    manifest = json.load(open(os.path.join(out, "parity_8_s2_manifest.json")))
    check("manifest holds the work map", manifest["workmap"] == FP_A and manifest["items_total"] == 10, str(manifest))
    rows = list(csv.DictReader(open([p for p in os.listdir(out) if p.endswith(".csv")][0] and os.path.join(out, [p for p in os.listdir(out) if p.endswith(".csv")][0]))))
    check("rows carry the fingerprint and the total", all(r["workmap"] == FP_A and r["items_total"] == "10" for r in rows) and len(rows) == 2, str(rows))
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", out, "--binary", binary])
    check("merge: complete run is ok against OEIS with mod-8", rc == 0 and "Q(8) = 92 ok (OEIS A000170)" in o and "PASS" in o, o)
    # 2. a binary with another work map is refused by the worker and records nothing
    rc, o = run([WORKER, "8", "--stride", "2", "--binary", binary, "--out", out, "--redo", "--tag", "other"], env={"FAKE_FP": FP_B})
    check("worker refuses a different work map", rc != 0 and "REFUSED" in o and "work map" in o, o)
    check("nothing recorded by the refused worker", not any("other" in p for p in os.listdir(out)), str(os.listdir(out)))
    # 3. mixed fingerprints across files are refused by the merge
    d = os.path.join(tmp, "mixed"); os.makedirs(d)
    write_csv(os.path.join(d, "parity_8_s2_w0of2_h1.csv"), [row(0, 46)])
    write_csv(os.path.join(d, "parity_8_s2_w1of2_h2.csv"), [row(1, 46, fp=FP_B, src="h2")])
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge refuses mixed work maps", rc != 0 and "INCOMPATIBLE work maps" in o and "no total can be trusted" in o, o)
    # 4. a manifest that disagrees with the records is refused
    d = os.path.join(tmp, "manifest"); os.makedirs(d)
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h1.csv"), [row(0, 46), row(1, 46)])
    json.dump({"n": 8, "stride": 2, "workmap": FP_B, "items_total": 10}, open(os.path.join(d, "parity_8_s2_manifest.json"), "w"))
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge refuses records that disagree with the manifest", rc != 0 and "INCOMPATIBLE" in o, o)
    # 5. missing phase → PARTIAL, missing file; conflict → CONFLICT
    d = os.path.join(tmp, "missing"); os.makedirs(d)
    write_csv(os.path.join(d, "parity_8_s2_w0of2_h1.csv"), [row(0, 46)])
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge reports a missing phase", rc != 0 and "MISSING 1 phases: 1" in o and open(os.path.join(d, "parity_8_s2_missing.txt")).read().strip() == "1", o)
    d = os.path.join(tmp, "conflict"); os.makedirs(d)
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h1.csv"), [row(0, 46), row(1, 46)])
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h1_verify.csv"), [row(0, 47)])
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge reports a conflict", rc != 0 and "CONFLICT phase 0" in o, o)
    # 6. agreeing duplicates count as verification
    d = os.path.join(tmp, "verified"); os.makedirs(d)
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h1.csv"), [row(0, 46), row(1, 46)])
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h2_verify.csv"), [row(0, 46, src="h2")])
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge counts agreeing duplicates as verification", rc == 0 and "1 phases verified by duplicates" in o, o)
    # 7. pre-fingerprint records: refused unless admitted
    d = os.path.join(tmp, "old"); os.makedirs(d)
    old_fields = ["n", "stride", "phase", "q_partial", "placements", "items", "wall_s", "finished_at", "host", "gpu", "binary_sha", "commit", "env"]
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h1.csv"), [row(0, 46), row(1, 46)], fields=old_fields)
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge refuses records without a fingerprint", rc != 0 and "without a work-map fingerprint" in o, o)
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d, "--allow-missing-fingerprint"])
    check("merge admits them only with --allow-missing-fingerprint", rc == 0 and "WARNING" in o and "ok (OEIS" in o, o)
    # 8. item counts that do not add up
    d = os.path.join(tmp, "items"); os.makedirs(d)
    write_csv(os.path.join(d, "parity_8_s2_w0of1_h1.csv"), [row(0, 46, items=5), row(1, 46, items=4)])
    rc, o = run([MERGE, "8", "--stride", "2", "--dir", d])
    check("merge reports item counts that do not add up", rc != 0 and "ITEMS MISMATCH" in o, o)
print(f"{'all tool tests passed' if fails == 0 else str(fails) + ' tool test(s) FAILED'}")
sys.exit(1 if fails else 0)
