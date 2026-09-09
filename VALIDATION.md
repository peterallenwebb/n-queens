# Validation

What is checked, by what, and what each check establishes. The aim is that a
reader can run everything on the CPU side in minutes on a laptop, the GPU side
in minutes on any NVIDIA GPU, and see the whole chain that the production count
relies on.

## 1. Tests in this repository (`cargo test --release`)

`tests/validation.rs`, CPU only, under a second:

| test | establishes |
|---|---|
| `reference_matches_published_values` | the independent brute-force solver agrees with OEIS A000170 for n ≤ 12 |
| `balance_lemma_holds_for_every_solution` | Lemma 2 of METHOD.md on every solution, n ≤ 11 (brute force) |
| `colex_ranks_are_a_bijection_per_popcount_class` | the table indexing is a bijection for every width ≤ 14 |
| `two_level_rank_table_matches_flat_ranks` | the shared-memory rank table the kernel uses equals the flat colex ranks for all 14-bit masks |
| `even_and_odd_bit_compressions` | the parity split of a column mask is invertible |
| `complement_ranks_invert_the_subset` | the complement map used by the join is right for every n ≤ 28, parity and k |
| `checkerboard_join_matches_reference_and_published_values` | the decomposition (CPU) equals brute force (n ≤ 11) and OEIS (n ≤ 13), with and without the 180° symmetry |
| `one_pair_matches_a_direct_count` | one enumerate-and-join pair equals the brute-force number of solutions whose even configuration has that row set, every pair of n = 8 and n = 9 |
| `strided_phases_and_ranges_add_up` | phases of a stride, and ranges, partition the work and their partials sum to Q(n) |
| `work_item_weights_cover_every_pair_exactly_once` | the symmetry weights account for every pair, n ≤ 16 |
| `work_map_fingerprints_are_stable` | golden fingerprints of the work map for n = 8, 12, 13 with and without symmetry; any change to the item ordering must update them consciously |
| `ticket_counts_are_checked` | ticket counts are checked arithmetic; an oversized prefix is refused rather than wrapped |
| `c2_count_matches_brute_force_and_the_identity_holds` | the mod-8 counter equals a brute-force count of the rotation-fixed solutions (n ≤ 12) and Q ≡ S₂ (mod 8) holds on the published values (n ≤ 22) |
| `checkerboard_join_n14_to_n18` (`--ignored`) | the CPU decomposition equals OEIS for n = 14 … 18 |

`tests/gpu.rs` (`cargo test --release --features cuda -- --ignored --test-threads 1`,
needs an NVIDIA GPU): the GPU count equals OEIS for n = 8 … 22; the phases of a
stride sum to Q(20); the alternate table configuration (flat ranks, u64
entries) and the run without the symmetry give the same count; the run reports
the largest table entry (98 at n = 20).

## 2. The ladder

Every GPU binary that produces a production count first runs
`nqueens-parity <n> --cuda` for n = 16 … 23 (seconds to a few minutes) and must
print `ok` (equal to OEIS A000170) and `mod-8 check: PASS` at every step. The
ladders run before each production launch are kept in `logs/ladders/`.

## 3. Production counts

Three checks apply to every full count:

1. **The published value.** Q(25) and Q(26) (and Q(27), in progress) are
   reproductions of known values, computed with exactly the machinery that
   computes Q(28): same kernel, same host code, same chunking and merge.
2. **The mod-8 identity** Q ≡ S₂ (mod 8) (METHOD.md), with S₂ counted by an
   unrelated program (`burnside.rs`). `nqueens-parity <n> --mod8 <count>` runs
   it on a merged total (551 s for n = 28 on one laptop core).
3. **Recomputation of a sample of phases** with a different configuration of
   the same decomposition (u64 tables and flat ranks instead of u32 tables and
   two-level shared-memory ranks; `--env NQ_T32=0 NQ_RANK2=0`), recorded as
   duplicate phase records that `tools/merge.py` reports as
   "verified by duplicates" and refuses if they disagree. For Q(26), chunks 23
   and 26 of 27 were recomputed this way (`logs/q26/verification_reruns.log`).

A fleet run adds **provenance and compatibility per phase**: host, GPU
model, SHA-256 of the binary, source commit, the NQ_* environment and the
**work-map fingerprint** (a hash of n, the symmetry setting and the ordered
work-item list the phases are cut from) are recorded with every phase in the
CSVs (`logs/`). The merge requires one fingerprint across all records and the
run manifest, every phase 0 … S−1 exactly once with duplicates agreeing, and
phase item counts that add up to the work map's total; a worker refuses to
start or to record a phase against a manifest with another fingerprint.
`tools/test_tools.py` exercises the refusals against a fake binary.

## 4. Field sizes at n = 28

The kernel refuses n > 28 at compile time: column masks occupy bits 0–27 of a
u32 and the frame's row index rides in bits 28–31 of two words. Every diagonal
shift has a row gap ≤ 27 and is masked to the board before the row nibble is
packed. The two-level colex ranks split a 14-bit one-parity column mask into 7 + 7 bits,
exactly the 14 columns per parity at n = 28 (the host refuses more). Ticket
counts are computed with checked arithmetic before any device work and a
prefix giving more than 2⁴⁰ tickets for one partition is refused, so an
oversized `NQ_PREF` fails instead of wrapping and skipping tickets (test
`ticket_counts_are_checked`). Tickets, table offsets, table indices,
per-thread accumulators, pair sums and join products are u64; the host total is u128 and is checked against u64 before it
is printed (Q(28) ≈ 2.5 × 10¹⁸ < 1.8 × 10¹⁹). Table entries are u32; the largest
entries observed with `NQ_TABMAX=1` are 98 (n = 20), 230 (n = 22), 818 (n = 24,
a 1 % sample), 967 (n = 26, 101 items), 3,662 (n = 28, the 101 heaviest-sampled
items) — growing ×3–4 per two steps of n, five orders of magnitude below 2³².
The per-lane stack must hold L − min(prefix, L) frames for a partial configuration with L rows;
L varies per work item (it is any even number for even n), the default of
n − prefix + 1 frames covers the worst case, and the host refuses a smaller
`NQ_MAXSTACK` after checking every item of the run.

## 5. What the chain does not cover

The checks establish that the program computes the identity of METHOD.md
correctly on every case with a reference value, that its parts agree with
independent brute force, and that the production count passes an independent
congruence and a partial recomputation. They do not guard against a hardware
fault that corrupts a phase without changing its exit status; the duplicate
phases across machines (check 3) are the mitigation, and a larger verification
sample can be run at will with `tools/worker.py --redo --tag verify`.
