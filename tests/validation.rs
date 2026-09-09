//! Validation of the balanced checkerboard join on the CPU (VALIDATION.md). Everything here runs
//! without a GPU: `cargo test --release`. The GPU tests are in tests/gpu.rs (feature `cuda`, ignored
//! by default).
use nqueens_parity::burnside::{count_c2, verify_mod8};
use nqueens_parity::geometry::{binom, colex_rank, even_bits, odd_bits, Geometry};
use nqueens_parity::known::{known, KNOWN_Q};
use nqueens_parity::parity::{count_cpu, enumerate, join, work_items, PartSpec, Selection};
use nqueens_parity::reference;

const THREADS: usize = 4;

// ---------- the independent reference

#[test]
fn reference_matches_published_values() {
    for n in 1..=12 { assert_eq!(reference::count(n) as u128, KNOWN_Q[n as usize - 1], "n = {n}"); }
}

/// The balance lemma (METHOD.md): in every solution the number of queens on (even row, even column)
/// squares equals the number on (odd row, odd column) squares for even n, and exceeds it by one for
/// odd n. It follows from counting even rows and odd columns, and it is what lets the count skip
/// every unbalanced row set.
#[test]
fn balance_lemma_holds_for_every_solution() {
    for n in 4..=11 {
        let mut checked = 0u64;
        reference::for_each_solution(n, |p| {
            let k = reference::class_counts(p);
            let expected_gap = (n % 2) as u32;
            assert_eq!(k[0], k[3] + expected_gap, "n = {n}, solution {p:?}: k00 = {}, k11 = {}", k[0], k[3]);
            assert_eq!(k[1], k[2], "n = {n}: k01 = {}, k10 = {}", k[1], k[2]);
            checked += 1;
        });
        assert_eq!(checked as u128, KNOWN_Q[n as usize - 1]);
    }
}

// ---------- geometry

#[test]
fn colex_ranks_are_a_bijection_per_popcount_class() {
    for w in 1..=14u32 {
        let g_full = (1u32 << w) - 1;
        for k in 0..=w {
            let mut seen = vec![false; binom(w, k) as usize];
            for m in 0..=g_full {
                if m.count_ones() != k { continue; }
                let r = colex_rank(m) as usize;
                assert!(r < seen.len() && !seen[r], "w = {w}, k = {k}, mask {m:#b}");
                seen[r] = true;
            }
            assert!(seen.iter().all(|&s| s));
        }
    }
}

#[test]
fn two_level_rank_table_matches_flat_ranks() {
    let t = Geometry::two_level_rank_table();
    for m in 0..(1u32 << 14) { assert_eq!(Geometry::two_level_rank(&t, m) as u64, colex_rank(m), "mask {m:#b}"); }
}

#[test]
fn even_and_odd_bit_compressions() {
    assert_eq!(even_bits(0b1010_1010), 0);
    assert_eq!(even_bits(0b0101_0101), 0b1111);
    assert_eq!(odd_bits(0b1010_1010), 0b1111);
    assert_eq!(even_bits((1 << 26) | 1), (1 << 13) | 1);
    assert_eq!(odd_bits(1 << 27), 1 << 13);
    for m in 0..(1u32 << 16) {
        let (e, o) = (even_bits(m), odd_bits(m));
        let mut back = 0u32;
        for i in 0..16 { if e >> i & 1 == 1 { back |= 1 << (2 * i); } if o >> i & 1 == 1 { back |= 1 << (2 * i + 1); } }
        assert_eq!(back, m);
    }
}

#[test]
fn complement_ranks_invert_the_subset() {
    for n in 2..=28u32 {
        let g = Geometry::new(n);
        for which in 0..2usize {
            let cnt = if which == 0 { g.ne } else { g.no };
            let fullw = (1u32 << cnt) - 1;
            for k in 0..=cnt {
                let comp = g.complement_ranks(which, k);
                for (rank, &m) in g.unrank[which][k as usize].iter().enumerate() {
                    assert_eq!(g.rank_tab[m as usize] as usize, rank);
                    assert_eq!(g.unrank[which][(cnt - k) as usize][comp[rank] as usize], fullw ^ m, "n = {n}, parity {which}, k = {k}");
                }
            }
        }
    }
}

// ---------- the decomposition

#[test]
fn checkerboard_join_matches_reference_and_published_values() {
    for n in 1..=13u32 {
        let st = count_cpu(n, THREADS, false, &Selection::ALL);
        assert_eq!(st.q, KNOWN_Q[n as usize - 1], "n = {n} with symmetry");
        if n <= 11 { assert_eq!(st.q, reference::count(n) as u128, "n = {n} vs brute force"); }
        let st2 = count_cpu(n, THREADS, true, &Selection::ALL);
        assert_eq!(st2.q, KNOWN_Q[n as usize - 1], "n = {n} without symmetry");
    }
}

/// A single enumerate-and-join pair, checked against a direct count: for n = 8 (even: the partner is the
/// even configuration on the reflected complementary rows) and n = 9 (odd: the partner is the odd
/// configuration on the complementary rows), enumerate both partial configurations of every pair and
/// compare the joined sum with the brute-force number of solutions whose even configuration has exactly
/// the row set of the pair's even configuration.
#[test]
fn one_pair_matches_a_direct_count() {
    for n in [8u32, 9] {
    let g = Geometry::new(n);
    for it in work_items(&g, true) {
        let a = PartSpec::new(&g, it.r, true);
        let b = PartSpec::new(&g, it.partner, n % 2 == 0);
        let mut ta = vec![0u64; a.size()];
        let mut tb = vec![0u64; b.size()];
        enumerate(&g, &a, &mut ta);
        enumerate(&g, &b, &mut tb);
        let (sum, _) = join(&g, &a, &ta, &b, &tb);
        let mut direct = 0u128;
        reference::for_each_solution(n, |p| {
            let even_rows_of_solution: u32 = p.iter().enumerate().filter(|(r, &c)| (*r as u32 + c as u32) % 2 == 0).map(|(r, _)| 1u32 << r).sum();
            if even_rows_of_solution == it.r { direct += 1; }
        });
        assert_eq!(sum, direct, "n = {n}, row set {:#b}", it.r);
    }
    }
}

/// Phases of a stride partition the work items, so their partial counts add up to Q(n); ranges
/// [from, to) likewise. This is the property the fleet run relies on (RUNBOOK.md).
#[test]
fn strided_phases_and_ranges_add_up() {
    for (n, stride) in [(10u32, 3usize), (12, 5), (13, 4)] {
        let mut total = 0u128;
        let mut items = 0usize;
        for phase in 0..stride {
            let st = count_cpu(n, THREADS, false, &Selection { from: 0, to: usize::MAX, stride, phase });
            total += st.q;
            items += st.items;
        }
        let full = count_cpu(n, THREADS, false, &Selection::ALL);
        assert_eq!(total, full.q, "n = {n}, stride {stride}");
        assert_eq!(items, full.items_total);
        let mid = full.items_total / 3;
        let a = count_cpu(n, THREADS, false, &Selection { from: 0, to: mid, stride: 1, phase: 0 });
        let b = count_cpu(n, THREADS, false, &Selection { from: mid, to: usize::MAX, stride: 1, phase: 0 });
        assert_eq!(a.q + b.q, full.q);
    }
}

#[test]
fn work_item_weights_cover_every_pair_exactly_once() {
    // With the symmetry, the weights of the representatives sum to the number of pairs listed without it.
    for n in 4..=16u32 {
        let g = Geometry::new(n);
        let with: u64 = work_items(&g, false).iter().map(|it| it.weight as u64).sum();
        let without: u64 = work_items(&g, true).iter().map(|it| it.weight as u64).sum();
        assert_eq!(with, without, "n = {n}");
    }
}

// ---------- the mod-8 check

#[test]
fn c2_count_matches_brute_force_and_the_identity_holds() {
    for n in 1..=12u32 {
        assert_eq!(count_c2(n), reference::count_c2_brute(n), "n = {n}");
    }
    for n in 1..=22u32 {
        let (_, pass) = verify_mod8(n, known(n).unwrap());
        assert!(pass, "Q({n}) ≡ S2 (mod 8) failed on the published value");
    }
    // a wrong value is caught whenever it is off by a non-multiple of 8
    assert!(!verify_mod8(8, 93).1);
}

// ---------- slower checks: cargo test --release -- --ignored

#[test]
#[ignore]
fn checkerboard_join_n14_to_n18() {
    for n in 14..=18u32 {
        let st = count_cpu(n, std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4), false, &Selection::ALL);
        assert_eq!(st.q, KNOWN_Q[n as usize - 1], "n = {n}");
    }
}
