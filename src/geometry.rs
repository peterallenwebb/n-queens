//! Board geometry shared by the CPU and GPU paths: binomials, the colex ranking of column subsets,
//! and the even/odd bit compressions that split a column mask into its two parity halves.
//!
//! Columns of one parity are numbered by their position among the columns of that parity: the even
//! columns 0, 2, 4, … become bits 0, 1, 2, … of the "even mask" (`even_bits`), and likewise for the
//! odd columns. A configuration with `k` queens on even columns has its even-column subset ranked in
//! colex order among all `k`-subsets, so a dense table of size C(ne, ke) × C(no, ko) indexes every
//! (even subset, odd subset) pair a configuration can occupy.

/// Binomial coefficient C(a, b) (0 when b > a); exact in u64 for the sizes used here (a ≤ 32).
pub fn binom(a: u32, b: u32) -> u64 {
    if b > a { return 0; }
    let mut r = 1u64;
    for i in 0..b as u64 { r = r * (a as u64 - i) / (i + 1); }
    r
}

/// Colex rank of `mask` among all masks with the same popcount: Σ_i C(p_i, i) over the set bits
/// p_1 < p_2 < … (1-based i). Ranks of the k-subsets of an m-set are exactly 0 .. C(m, k) − 1.
pub fn colex_rank(mut mask: u32) -> u64 {
    let mut r = 0u64;
    let mut i = 1u32;
    while mask != 0 {
        let p = mask.trailing_zeros();
        r += binom(p, i);
        mask &= mask - 1;
        i += 1;
    }
    r
}

/// Compress the even-position bits (0, 2, 4, …) of `x` into the low 16 bits.
pub fn even_bits(mut x: u32) -> u32 {
    x &= 0x5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF;
    x
}

/// Compress the odd-position bits (1, 3, 5, …) of `x` into the low 16 bits.
pub fn odd_bits(x: u32) -> u32 { even_bits(x >> 1) }

/// Column-subset ranking tables for an n × n board.
pub struct Geometry {
    pub n: u32,
    /// mask of all n columns (and of all n rows)
    pub full: u32,
    /// number of even columns ⌈n/2⌉ and odd columns ⌊n/2⌋
    pub ne: u32,
    pub no: u32,
    /// colex rank within its popcount class, for every mask below 2^max(ne, no)
    pub rank_tab: Vec<u32>,
    /// unrank[which (0 even, 1 odd)][k][rank] → mask of the k-subset with that rank
    pub unrank: Vec<Vec<Vec<u32>>>,
}

impl Geometry {
    pub fn new(n: u32) -> Self {
        assert!((1..=28).contains(&n), "n must be in 1..=28 (masks are u32 and the GPU frame packs the row index above bit 27)");
        let ne = (n + 1) / 2;
        let no = n / 2;
        let w = ne.max(no);
        let rank_tab: Vec<u32> = (0..1u32 << w).map(|m| colex_rank(m) as u32).collect();
        let mut unrank = vec![vec![Vec::new(); w as usize + 1]; 2];
        for (which, cnt) in [(0usize, ne), (1usize, no)] {
            for k in 0..=cnt {
                let mut v = vec![0u32; binom(cnt, k) as usize];
                for m in 0..1u32 << cnt {
                    if m.count_ones() == k { v[rank_tab[m as usize] as usize] = m; }
                }
                unrank[which][k as usize] = v;
            }
        }
        Geometry { n, full: (1u32 << n) - 1, ne, no, rank_tab, unrank }
    }

    /// Reverse the n-bit mask: the image of a row set (or column set) under a reflection.
    pub fn reflect(&self, m: u32) -> u32 { m.reverse_bits() >> (32 - self.n) }

    /// Mask of the even rows 0, 2, 4, …
    pub fn even_rows(&self) -> u32 { self.full & 0x5555_5555 }

    /// For the k-subsets of the `which` columns (0 even, 1 odd): rank → rank of the complementary
    /// (cnt − k)-subset. This is how an even configuration's column subset finds the odd configuration's.
    pub fn complement_ranks(&self, which: usize, k: u32) -> Vec<u32> {
        let cnt = if which == 0 { self.ne } else { self.no };
        let fullw = (1u32 << cnt) - 1;
        self.unrank[which][k as usize].iter().map(|&m| self.rank_tab[(fullw ^ m) as usize]).collect()
    }

    /// Two-level colex rank table for masks of at most 14 bits, as the GPU kernel reads it from
    /// shared memory: `lo[m & 127]` is the rank of the low 7 bits, and `hi[m >> 7][popcount(m & 127)]`
    /// is the contribution of the high 7 bits given how many low bits precede them
    /// (Σ C(7 + p, c + i) over the set high bits p, 1-based i). rank(m) = lo + hi. Layout:
    /// 128 entries of lo followed by 128 × 8 entries of hi. Validated against `colex_rank` in the tests.
    pub fn two_level_rank_table() -> Vec<u32> {
        let mut v = vec![0u32; 128 + 128 * 8];
        for m in 0..128u32 { v[m as usize] = colex_rank(m) as u32; }
        for h in 0..128u32 {
            for c in 0..8u32 {
                let mut r = 0u64;
                let mut i = 1u32;
                let mut x = h;
                while x != 0 {
                    let p = x.trailing_zeros();
                    r += binom(7 + p, c + i);
                    x &= x - 1;
                    i += 1;
                }
                v[128 + (h as usize) * 8 + c as usize] = r as u32;
            }
        }
        v
    }

    /// Rank of a mask (≤ 14 bits) through the two-level table; mirrors the kernel's RANK macro.
    pub fn two_level_rank(table: &[u32], m: u32) -> u32 {
        table[(m & 127) as usize] + table[128 + ((m >> 7) as usize) * 8 + (m & 127).count_ones() as usize]
    }
}
