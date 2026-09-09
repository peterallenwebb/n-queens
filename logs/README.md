# Records

Every computation made with this program leaves its records here: the ladder
that validated the binary, the per-chunk or per-phase CSV with partial counts
and provenance, the driver log, the merge summary, the mod-8 check and the
verification reruns. Machines are rented Vast.ai boxes with one NVIDIA RTX 5090
(driver 580.159.03, CUDA 12.8, Ubuntu 22.04, 256-core hosts) unless noted.

## ladders/

`ladder_2026-09-09_5090.log` — n = 16 … 23 on the development repository's
production build (commit 882be62 there), run before the Q(27) launch: all
`ok`, all `mod-8 check: PASS`; n = 23 in 216.9 s.

`ladder_2026-09-09_new-crate_5090.log` — the same box, this repository's
binary: n = 8 … 22 all `ok` with mod-8 PASS, the GPU test suite (ladder,
phases of a stride, the alternate table configuration, the table-maximum
report), and the n = 22 table maximum, run while the Q(27) worker shared the
GPU (walls are inflated).

`ladder_2026-09-09_6b105ed_5090.log` — commit 6b105ed (the review items: work-map
fingerprint and manifest, checked ticket counts, fail-loud scripts): ladder
n = 16, 18, 20, 21 exact with the fingerprint line, an oversized prefix refused
before any device work, a two-worker fleet run at n = 20 (stride 4) through
`tools/worker.py` and `tools/merge.py` with the manifest, and the CPU and tool
test suites on the box.

## q26/  — Q(26) = 22,317,699,616,364,044 (equals OEIS A000170)

Computed 2026-09-09, 01:23–07:31 UTC, on one RTX 5090 by the development
repository's build (commit 882be62 there; kernel identical to
`kernel/parity.cu` apart from diagnostic branches that are compiled out in
the production configuration and were removed here). 27 contiguous chunks of
100,000 work items, driven by the range-chunk driver that preceded
`tools/worker.py`.

- `chunks.csv` — from, to, partial count, placements, wall, finish time per
  chunk; the partials sum to Q(26); Σ walls = 22,075 s = 6.13 h.
- `driver.log` — the driver's output, the box's GPU/driver/OS/toolchain.
- `verification_reruns.log` — chunks 23 (359 s) and 26 (50 s) recomputed with
  u64 tables and flat ranks (`NQ_T32=0 NQ_RANK2=0`): both partials reproduced
  exactly.
- mod-8: `nqueens-parity 26 --mod8 22317699616364044` → PASS (Q ≡ 4 mod 8,
  115,718,268 solutions fixed by the 180° rotation).

## q27/ — partial run, to be rerun on the fleet with the finalized code

Started 2026-09-09 07:49 UTC on the same box by the development repository's
build (commit 882be62 there) through the strided driver, stride 200, worker 0
of 1; stopped by the user at 18:31 UTC after 24 phases (0–23) to be restarted
as the final fleet test before Q(28). `partial_dev_build_phases_0-23.csv`
holds the 24 records: 1,594 s per phase on average, 9.5 × 10¹³ placements per
phase (5.96 × 10¹⁰ per second, 0.081 placements per solution), partial sum
28,186,222,081,928,376. The rerun with the finalized code cuts the same work
map (the item ordering is unchanged; the finalized binary's fingerprint for
n = 27 is cb51fbf2a3a51f62), so these 24 partials must be reproduced exactly
by its phases 0–23 — a cross-check between the two builds on top of the
published value.

## q28/ — to come
