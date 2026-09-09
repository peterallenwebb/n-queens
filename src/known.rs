//! Published values of Q(n) = number of solutions of the n-queens problem, OEIS A000170,
//! n = 1..27 (index n − 1). Q(27) is the Preußer–Engelhardt (2016) value, confirmed by later
//! independent computations. Used by the tests and by the run ladder; a run at an n beyond this
//! table prints its count without a reference.
pub const KNOWN_Q: [u128; 27] = [
    1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104,
    666090624, 4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440,
    227514171973736, 2207893435808352, 22317699616364044, 234907967154122528,
];

/// The published count for `n`, if any.
pub fn known(n: u32) -> Option<u128> {
    if n == 0 { None } else { KNOWN_Q.get(n as usize - 1).copied() }
}
