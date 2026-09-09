# The Balanced Checkerboard Join

This note describes the identity the program uses to compute Q(n) and proves the two simple lemmas it rests on.

## Setting

An n-queens solution, which we call a configuration, is a placement of n mutually non-attacking queens on an n × n board: one queen per row, one per column, no two on a common diagonal. The total count of such configurations is Q(n) and its known values are recorded in OEIS A000170. Rows and columns are numbered 0…n−1. The *parity* of a square (r, c) is the parity of r + c: the square is **even** or **odd**. The four *classes* of a square are its (row parity, column parity): even = (0,0) ∪ (1,1), odd = (0,1) ∪ (1,0).

## Lemma 1 (Diagonals Have Constant Parity)

This self-evident lemma was mentioned in passing by Simkin (§4.2 of "The number of n-queens configurations": diagonals of a chessboard intersect only when they have the same colour) and follows directly from the definition of diagonals as squares with identical r + c or r − c.

## Lemma 2 (Balance)

Let a solution have k₀₀, k₀₁, k₁₀, k₁₁ queens in the four classes (k_ab = queens in rows ≡ a and columns ≡ b mod 2). Then

    k₀₀ − k₁₁ = ⌈n/2⌉ − ⌊n/2⌋ = n mod 2
and

    k₀₁ = k₁₀.

*Proof.* Every row holds exactly one queen, so k₀₀ + k₀₁ = number of even rows = ⌈n/2⌉. Every column holds exactly one queen, so k₀₁ + k₁₁ = number of odd columns = ⌊n/2⌋. Subtracting gives the first identity; the second follows the same way from the odd rows and the even columns. ∎

## Even and odd configurations

An **even configuration** with row set R ⊆ {0, …, n−1} and column set C is a placement of |R| = |C| mutually non-attacking queens on even squares, one in each row of R and one in each column of C: a partial solution confined to the even squares (in Simkin's terms, a partial n-queens configuration restricted to one parity). E[R, C] denotes the number of even configurations with row set R and column set C; O[R, C] is the same for **odd configurations**, confined to the odd squares. On an even square, a queen in an even row sits in an even column and one in an odd row in an odd column, so an even configuration with row set R uses exactly kₑ = |R ∩ even rows| even columns and kₒ = |R ∩ odd rows| odd columns; for an odd configuration the roles are swapped.

## The identity

**Theorem.** With R̄ and C̄ the complements of R and C in {0, …, n−1},

    Q(n) = Σ_R Σ_C E[R, C] · O[R̄, C̄].

*Proof.* The queens of a configuration split into those on even squares and those on odd squares. The former are an even configuration with some row set R and column set C (they are mutually non-attacking as a subset of a solution); the latter are an odd configuration, and since each row and each column holds exactly one queen, its row set is R̄ and its column set is C̄. Conversely, take any even configuration (R, C) and any odd configuration (R̄, C̄). Together they place n queens, one per row and one per column. Two queens of the same parity do not attack (each partial configuration is non-attacking); two queens of different parity share no row and no column (the row and column sets are disjoint) and no diagonal (Lemma 1). So the union is a solution, and the two constructions are inverse to each other. Summing over (R, C) counts every solution exactly once. ∎

Lemma 2 says which terms can be nonzero: in a solution, the queens on even squares occupy k₀₀ even rows and k₁₁ odd rows, so the row set R of the even configuration of every solution satisfies |R ∩ even rows| − |R ∩ odd rows| = n mod 2. For any other R the inner sum Σ_C E[R, C] · O[R̄, C̄] counts solutions whose even configuration has row set R, of which there are none. The program therefore enumerates only the **balanced** row sets. For n = 28 there are C(14,k)² such sets summed over k, some 40 million, before the symmetries below.

## Symmetries Used

*Row reflection ρ*: (r, c) ↦ (n−1−r, c). For **even** n, n−1 is odd, so ρ flips the row parity and hence the square parity: it maps odd configurations with row set S and column set C to even configurations with row set ρ(S) and column set C, bijectively. Hence O[R̄, C̄] = E[ρ(R̄), C̄] and only even configurations are ever enumerated:

    Q(n) = Σ_R Σ_C E[R, C] · E[ρ(R̄), C̄]        (n even).

The partner of R is P = ρ(R̄), and the partner of P is ρ(P̄) = ρ(ρ(R)) = R, so the terms come in unordered pairs {R, P}; the program lists each unordered pair once with weight 2 (weight 1 when R = P). For **odd** n, ρ preserves square parity, so even and odd configurations are both enumerated: the pair is (R even, R̄ odd).

*Rotation σ by 180°*: (r, c) ↦ (n−1−r, n−1−c) preserves square parity and maps solutions to solutions; it maps a configuration with row set R and column set C to one with row set ρ(R) and column set ρ(C) of the same parity, so E[ρ(R), ρ(C)] = E[R, C]. Applied to the sum it maps the term of the pair {R, P} to the term of the pair {ρ(R), ρ(P)} (even n) or of the row set ρ(R) (odd n) with the same value, so one representative per σ-orbit suffices, with the orbit size (1 or 2) folded into the weight. `work_items` in `parity.rs` lists the representatives and weights; the test `work_item_weights_cover_every_pair_exactly_once` checks that the weights add up to the number of pairs without the symmetry, and every count is also checked with the symmetry switched off (`--nosym`).

## The Computation

**Partition by row set.** Each row set R is one bounded depth-first search: one queen per row of R on the columns of the right parity, with the usual column and diagonal masks (the diagonal masks are shifted by the gap to the next row of R). Every leaf is a configuration with some column set C; it is counted into a dense table indexed by (colex rank of C ∩ even columns among kₑ-subsets, colex rank of C ∩ odd columns among kₒ-subsets), of size C(⌈n/2⌉, kₑ) · C(⌊n/2⌋, kₒ).

**Join.** For a pair, the sum Σ_C E[R, C] · E[P, C̄] walks one table and reads the other at the complementary ranks (`complement_ranks` in `geometry.rs`), a precomputed map from the rank of a subset to the rank of its complement. On the GPU the larger side is never stored: its configurations are enumerated and, at each leaf, the partner's table entry at the complementary ranks is added to the pair's sum ("read-side join").

**Work items and phases.** The list of (representative, partner, weight) items is ordered heaviest first. A run covers a *selection* of it: the index range [from, to) thinned to the indices ≡ p (mod S). The S phases of one stride partition the list, so their partial sums add up to Q(n) (test `strided_phases_and_ranges_add_up`); a phase samples heavy and light items alike, so phases cost about the same and any number of machines can take disjoint phase sets with no coordination (RUNBOOK.md).

**GPU details.** Items are processed in batches limited by a table budget. Phase 1 enumerates the smaller-table side of each item into its table; phase 2 enumerates the partner side with read-side joins; items whose partner is the same row set (even n, R = ρ(R̄)) are joined by a separate kernel over the table with itself. Within a kernel, the first few rows of every configuration are decided by a *ticket* (a mixed-radix number choosing one column of the right parity in each prefix row; dead prefixes are skipped), so the search splits into millions of roots that persistent threads pull from a global counter; below the prefix the search runs from a per-lane stack in shared memory. Table entries are u32 (the largest entry observed is 3,662 at n = 28 on a sample; see VALIDATION.md), and all offsets, tickets, accumulators and pair sums are u64; the host accumulates the weighted total in u128.

## Cost

Configurations are far fewer than solutions at the sizes of interest, so the search does less work than a direct enumeration of solutions. Measured placements per solution (search steps, including the enumeration of both sides): 0.455 at n = 22, 0.179 at n = 24, 0.0665 at n = 26 (even n), 0.590 at n = 23 and 0.226 at n = 25 (odd n); the ratio falls by about ×0.4 per two steps of n. On one RTX 5090: Q(24) in 519 s, Q(25) in 9,422 s, Q(26) in 22,075 s.

## The mod-8 Check

The dihedral group of the square acts on solutions. No solution is fixed by a reflection: a solution fixed by the reflection in a vertical or horizontal axis would need every queen on the axis; one fixed by the reflection in the main diagonal would be a permutation equal to its inverse, and each of its 2-cycles (r ↦ c, c ↦ r) would put queens on (r, c) and (c, r), which share the anti-diagonal r + c, while two fixed points would share the main diagonal, so n ≤ 1; the anti-diagonal is the same argument. Hence every orbit has size 8, 4 (fixed by the 180° rotation only) or 2 (fixed by the 90° rotation). With k₈, k₄, k₂ orbits of each size and S₂ the number of solutions fixed by the 180° rotation,

    Q = 8k₈ + 4k₄ + 2k₂,   S₂ = 4k₄ + 2k₂
  
so

    Q ≡ S₂ (mod 8).

`burnside.rs` counts S₂ directly (each queen in the first half of the columns forces its partner) in minutes for n = 28, and every full count is checked against it. The test `c2_count_matches_brute_force_and_the_identity_holds` compares the counter with a brute-force count of the fixed solutions and checks the identity on the published values.
