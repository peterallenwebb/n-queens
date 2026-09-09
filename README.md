# Computing Q(n) by a Balanced Checkerboard Join

This repository contains code for counting n-queens solutions by a "balanced checkerboard join," a decomposition of every solution into its queens on even squares (row + column even) and its queens on odd squares (METHOD.md). The repository also includes tests that validate the counting code and records of the computations made with it. The goal is to compute Q(28) on a fleet of worker nodes equipped with NVIDIA RTX 5090 GPUs, and is published so that the computation can be examined and reproduced.

**Current Status:** 
* The known values of Q(25) and Q(26) have been reproduced (one RTX 5090, 9,422s and 22,075s).
* A reproduction of Q(27) is in progress
* Every count is checked against the published value where one exists, against the mod-8 identity Q ≡ S₂ (mod 8) with S₂ counted by an unrelated program, and by recomputing a sample of its phases in a different configuration (VALIDATION.md).

## Method Sketch

Along any diagonal the parity of row + column is constant, so queens on even squares and queens on odd squares can attack each other only through rows and columns. A configuration (a complete solution) is therefore exactly an even configuration (a partial solution confined to the even squares) and an odd configuration on the complementary rows and columns, and

    Q(n) = Σ_R Σ_C E[R, C] · O[R̄, C̄],

where E[R, C] (O[R, C]) is the number of even (odd) configurations with row set R and column set C. Only balanced row sets contribute (Lemma 2 of METHOD.md). Row reflection turns odd configurations into even ones for even n, and the 180° rotation cuts the work in half. Each row set is one bounded search whose leaves are counted into a dense table indexed by the ranks of the configuration's even and odd column subsets; pairs are joined by reading the partner's table at the complementary ranks. Importantly, partial configurations are far fewer than solutions at n ≥ 20: the search costs 0.0665 steps per solution at n = 26.

## Layout

| path | contents |
|---|---|
| `src/geometry.rs` | Binomials, colex ranks of column subsets, parity bit compressions, the two-level rank table the kernel uses |
| `src/parity.rs` | Work items (row-set pairs with symmetry weights), the CPU enumerator of even and odd configurations, the join, the CPU count, selections (ranges and phases) |
| `src/gpu.rs`, `kernel/parity.cu` | The CUDA production path (feature `cuda`): batches, tickets, the stack-resident enumeration, the read-side join |
| `src/burnside.rs` | The 180°-symmetric count and the mod-8 check |
| `src/reference.rs` | A reference brute-force solver (tests only) |
| `src/known.rs` | Known values of Q(n), 0 ≤ n ≤ 27,  per OEIS A000170 |
| `src/main.rs` | The `nqueens-parity` command |
| `tests/` | Validation code (see VALIDATION.md) |
| `tools/` | Fleet operation: `worker.py`, `merge.py`, `ship.sh`, `start_worker.sh`, `collect.sh` (RUNBOOK.md) |
| `logs/` | Records of the computations: ladders, chunk and phase CSVs with provenance, merge summaries, checks |

## Build, Test, and Run

```
cargo build --release                      # CPU only
cargo test --release                       # the validation suite, under a second
cargo test --release -- --ignored          # plus n = 14 … 18 on the CPU, seconds
cargo build --release --features cuda      # the GPU path (NVIDIA driver + libnvrtc)
cargo test --release --features cuda -- --ignored --test-threads 1      # GPU ladder n = 8 … 22, phases, configurations

./target/release/nqueens-parity 16 --cpu           # 14772512 ok, mod-8 PASS
./target/release/nqueens-parity 22 --cuda          # 2691008701644 ok in 13.4 s on an RTX 5090
./target/release/nqueens-parity 12 --reference     # brute force
./target/release/nqueens-parity 26 --mod8 22317699616364044
NQ_PAIR_STRIDE=200 NQ_PAIR_PHASE=17 ./target/release/nqueens-parity 27 --cuda    # one phase of a fleet run
```

The CPU path is independent of the GPU; it counts n = 22 in about two hours on a MacBook Pro M1. The GPU kernel is compiled at run time by NVRTC for the device found, so any recent NVIDIA GPU can run the ladder.

## Records

`logs/` keeps, for each computation, the chunk or phase CSV (one row per unit of work with its partial count, wall time and provenance), the driver log, the merge summary, the mod-8 output and the verification reruns, with a README describing the machine, driver and source revision. See `logs/README.md`.

## Provenance

The decomposition, kernel and tools were developed in a larger research repository; this repository contains the production subset, re-validated from scratch (the tests above and the GPU ladder in `logs/ladders/`). The Q(25) and Q(26) records in `logs/` were produced by that development repository's build of the same kernel, as noted in `logs/README.md`; every later computation is run from this repository at a recorded commit.

## License

MIT (see LICENSE).
