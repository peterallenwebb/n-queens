//! The mod-8 consistency check every full count must pass.
//!
//! The dihedral group of the square acts on solutions. No solution is fixed by a reflection (a
//! solution fixed by the reflection in a diagonal would be a permutation equal to its inverse whose
//! 2-cycles put two queens on one diagonal; the other reflections are ruled out likewise), so every
//! orbit has size 8 (no symmetry), 4 (fixed by the 180° rotation only) or 2 (fixed by the 90°
//! rotation). With k8, k4, k2 orbits of each size and S2 = number of solutions fixed by the 180°
//! rotation,
//!
//!   Q = 8·k8 + 4·k4 + 2·k2,   S2 = 4·k4 + 2·k2,   so   Q ≡ S2 (mod 8).
//!
//! S2 is cheap to count (each queen in the first half of the columns forces its partner), so a
//! claimed Q gets an independent 3-bit check. The check is weak on its own (a random wrong value
//! passes with probability 1/8) but it is independent of the decomposition, and it is one of three
//! checks applied to every production count (VALIDATION.md).

/// Number of solutions fixed by the 180° rotation: place columns 0 .. n/2 − 1, each queen at (r, c)
/// forcing its partner at (n−1−r, n−1−c); odd n has the centre queen fixed first.
pub fn count_c2(n: u32) -> u64 {
    let (mut rook, mut add, mut sub) = (0u32, 0u64, 0u64);
    if n % 2 == 1 {
        let m = (n - 1) / 2;
        rook |= 1 << m;
        add |= 1u64 << (2 * m);
        sub |= 1u64 << (n - 1);
    }
    c2_rec(n, 0, rook, add, sub)
}

fn c2_rec(n: u32, c: u32, rook: u32, add: u64, sub: u64) -> u64 {
    if c == n / 2 { return 1; }
    let full = (1u32 << n) - 1;
    let mut m = full & !(rook | (add >> c) as u32 | (sub >> (n - 1 - c)) as u32);
    let mut total = 0;
    while m != 0 {
        let bit = m & m.wrapping_neg();
        m ^= bit;
        let r = bit.trailing_zeros();
        let (pr, pc) = (n - 1 - r, n - 1 - c);
        let pbit = 1u32 << pr;
        let padd = 1u64 << (pr + pc);
        let psub = 1u64 << (pr + n - 1 - pc);
        let rook1 = rook | bit;
        let add1 = add | 1u64 << (r + c);
        let sub1 = sub | 1u64 << (r + n - 1 - c);
        if rook1 & pbit != 0 || add1 & padd != 0 || sub1 & psub != 0 { continue; }
        total += c2_rec(n, c + 1, rook1 | pbit, add1 | padd, sub1 | psub);
    }
    total
}

/// Check a claimed count against the identity Q ≡ S2 (mod 8). Returns (S2, pass).
pub fn verify_mod8(n: u32, claimed: u128) -> (u64, bool) {
    let s2 = count_c2(n);
    (s2, claimed % 8 == (s2 % 8) as u128)
}
