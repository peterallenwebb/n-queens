//! An independent brute-force n-queens solver, used only by the tests to check the checkerboard
//! decomposition against something that has nothing to do with it: plain row-by-row backtracking
//! with column and diagonal bitmasks, visiting every solution.

/// Call `f` with each solution as the array of column indices by row.
pub fn for_each_solution(n: u32, mut f: impl FnMut(&[u8])) {
    assert!((1..=31).contains(&n));
    let full = (1u32 << n) - 1;
    let mut perm = vec![0u8; n as usize];
    fn go(n: u32, row: u32, cols: u32, ld: u32, rd: u32, full: u32, perm: &mut [u8], f: &mut impl FnMut(&[u8])) {
        if row == n { f(perm); return; }
        let mut free = full & !(cols | ld | rd);
        while free != 0 {
            let b = free & free.wrapping_neg();
            free ^= b;
            perm[row as usize] = b.trailing_zeros() as u8;
            go(n, row + 1, cols | b, ((ld | b) << 1) & full, (rd | b) >> 1, full, perm, f);
        }
    }
    go(n, 0, 0, 0, 0, full, &mut perm, &mut f);
}

/// Q(n) by brute force.
pub fn count(n: u32) -> u64 {
    let mut c = 0u64;
    for_each_solution(n, |_| c += 1);
    c
}

/// Number of solutions invariant under the 180° rotation (row r, column c) ↦ (n−1−r, n−1−c),
/// by brute force — the independent check of `burnside::count_c2`.
pub fn count_c2_brute(n: u32) -> u64 {
    let mut c = 0u64;
    for_each_solution(n, |p| {
        let m = n as usize - 1;
        if (0..p.len()).all(|r| p[m - r] as usize == m - p[r] as usize) { c += 1; }
    });
    c
}

/// Colour-class counts (k00, k01, k10, k11) of a solution: k_ab = queens in rows ≡ a, columns ≡ b (mod 2).
pub fn class_counts(perm: &[u8]) -> [u32; 4] {
    let mut k = [0u32; 4];
    for (r, &c) in perm.iter().enumerate() { k[((r & 1) << 1) | (c as usize & 1)] += 1; }
    k
}
