//! GPU validation (needs an NVIDIA GPU and `--features cuda`; all ignored by default):
//!     cargo test --release --features cuda -- --ignored --test-threads 1
#![cfg(feature = "cuda")]
use nqueens_parity::gpu::{run, GpuConfig};
use nqueens_parity::known::KNOWN_Q;
use nqueens_parity::parity::Selection;

fn cfg(n: u32) -> GpuConfig { GpuConfig::from_env(n).unwrap() }

#[test]
#[ignore]
fn gpu_ladder_matches_published_values() {
    for n in 8..=22u32 {
        let st = run(n, 0, false, &Selection::ALL, &cfg(n)).unwrap();
        assert_eq!(st.q, KNOWN_Q[n as usize - 1], "n = {n}");
        assert!(!st.partial);
    }
}

#[test]
#[ignore]
fn gpu_phases_add_up_and_configurations_agree() {
    let n = 20u32;
    let stride = 4;
    let mut total = 0u128;
    for phase in 0..stride {
        total += run(n, 0, false, &Selection { from: 0, to: usize::MAX, stride, phase }, &cfg(n)).unwrap().q;
    }
    assert_eq!(total, KNOWN_Q[n as usize - 1]);
    // the alternate table configuration (flat ranks, u64 entries) and no symmetry give the same count
    let mut alt = cfg(n);
    alt.rank2 = false;
    alt.t32 = false;
    assert_eq!(run(n, 0, true, &Selection::ALL, &alt).unwrap().q, KNOWN_Q[n as usize - 1]);
}

#[test]
#[ignore]
fn gpu_reports_the_largest_table_entry() {
    let n = 20u32;
    let mut c = cfg(n);
    c.tabmax = true;
    let st = run(n, 0, false, &Selection::ALL, &c).unwrap();
    assert_eq!(st.q, KNOWN_Q[n as usize - 1]);
    assert_eq!(st.table_max, Some(98));   // measured value (VALIDATION.md)
}
