//! The balanced checkerboard join: work items, the CPU configuration enumerator, the join, and the CPU count.
//!
//! Notation (METHOD.md). Squares are classed by (row parity, column parity); even = (0,0) ∪ (1,1),
//! odd = (0,1) ∪ (1,0). For a row set R and column set C let E[R, C] be the number of ways to place
//! |R| mutually non-attacking queens on even squares with row set exactly R and column set exactly
//! C (an "even configuration"); O[R, C] likewise on odd squares. Then
//!
//!   Q(n) = Σ_R Σ_C E[R, C] · O[R̄, C̄]           (R̄, C̄ = complements)
//!
//! because a solution is exactly an even configuration and an odd configuration on complementary rows and columns
//! (diagonals have constant parity, so the two configurations never attack each other diagonally).
//! For even n the row reflection ρ maps odd configurations to even configurations, so O[R̄, C̄] = E[ρ(R̄), C̄] and only
//! even configurations are enumerated; for odd n the two parities are enumerated separately. An even configuration with
//! row set R has k00 queens on (even row, even column) and k11 on (odd row, odd column); the numbers
//! of even and odd rows of R fix ke = k00 and ko = k11, and only balanced row sets occur in solutions
//! (k00 = k11 for even n, k00 = k11 + 1 for odd n) — see METHOD.md — so the others are skipped.
//!
//! The computation is partitioned by the row set: each R is one bounded depth-first search whose
//! leaves are counted into a dense table indexed by (colex rank of the even-column subset, colex rank
//! of the odd-column subset). A pair (R, partner) is joined by walking one table and reading the
//! partner's table at the complementary ranks. The 180° rotation of the board halves the work.
use crate::geometry::{binom, even_bits, odd_bits, Geometry};
use std::sync::atomic::{AtomicUsize, Ordering};

/// One single-parity configuration enumeration: the row set, the parity, and the table geometry.
#[derive(Clone, Debug)]
pub struct PartSpec {
    pub rowmask: u32,
    /// the rows of `rowmask` in increasing order
    pub rows: Vec<u32>,
    pub even: bool,
    /// numbers of even and odd columns this configuration uses (fixed by the row set and the parity)
    pub ke: u32,
    pub ko: u32,
    /// table dimensions C(ne, ke) and C(no, ko)
    pub width_even: u64,
    pub width_odd: u64,
}

impl PartSpec {
    pub fn new(g: &Geometry, rowmask: u32, even: bool) -> Self {
        let rows: Vec<u32> = (0..g.n).filter(|i| rowmask >> i & 1 != 0).collect();
        let ke_rows = (rowmask & g.even_rows()).count_ones();
        let ko_rows = rows.len() as u32 - ke_rows;
        // on even squares a queen in an even row sits on an even column and in an odd row on an odd column; odd squares are the reverse
        let (ke, ko) = if even { (ke_rows, ko_rows) } else { (ko_rows, ke_rows) };
        PartSpec { rowmask, rows, even, ke, ko, width_even: binom(g.ne, ke), width_odd: binom(g.no, ko) }
    }
    /// Number of table entries: one per (even subset, odd subset) pair of the right sizes.
    pub fn size(&self) -> usize { (self.width_even * self.width_odd) as usize }
    /// Column parity a configuration of this parity uses in row r.
    pub fn col_parity(&self, r: u32) -> u32 { if self.even { r & 1 } else { (r & 1) ^ 1 } }
}

/// A pair of configurations to enumerate and join: the even configuration's row set `r`, the partner row set, and the
/// symmetry weight (how many pairs of the full sum this representative stands for).
#[derive(Clone, Copy, Debug)]
pub struct WorkItem { pub r: u32, pub partner: u32, pub weight: u32 }

/// All work items of the count, in the order the chunking and the GPU batching use (heaviest first).
///
/// Even n: the partner of R is ρ(R̄) (its complement, reflected), so pairs {R, ρ(R̄)} are unordered;
/// one of the two orders is kept (weight 2 when the two differ, 1 when R = ρ(R̄)). With the 180°
/// rotation σ (which maps even configurations to even configurations and the pair {R, P} to {ρ(R), ρ(P)} up to the
/// same reflection bookkeeping), one representative per σ-orbit is kept with the orbit size folded
/// into the weight. Odd n: every balanced even row set R is paired with its complement R̄ (odd);
/// the rotation identifies R with its reflection ρ(R), so one of each pair is kept with weight 2.
/// `nosym` disables the rotation (every pair is listed, weights 1 or 2 only from the pair order),
/// which the tests use to confirm the symmetry accounting.
pub fn work_items(g: &Geometry, nosym: bool) -> Vec<WorkItem> {
    let full = g.full;
    let even_rows = g.even_rows();
    let odd_n = g.n % 2 == 1;
    let mut items = Vec::new();
    for r in 0..=full {
        let ke = (r & even_rows).count_ones();
        let ko = (r & !even_rows).count_ones();
        if odd_n {
            if ke != ko + 1 { continue; }
            let rr = g.reflect(r);
            if nosym { items.push(WorkItem { r, partner: full ^ r, weight: 1 }); }
            else if r < rr { items.push(WorkItem { r, partner: full ^ r, weight: 2 }); }
            else if r == rr { items.push(WorkItem { r, partner: full ^ r, weight: 1 }); }
        } else {
            if ke != ko { continue; }
            let p = g.reflect(full ^ r);
            if r > p { continue; }
            let base = if r == p { 1 } else { 2 };
            if nosym { items.push(WorkItem { r, partner: p, weight: base }); continue; }
            let k1 = r.min(p);
            let k2 = g.reflect(r).min(full ^ r);
            if k1 < k2 { items.push(WorkItem { r, partner: p, weight: base * 2 }); }
            else if k1 == k2 { items.push(WorkItem { r, partner: p, weight: base }); }
        }
    }
    // Longest-processing-time order: the widest tables (ke near the middle) first.
    items.sort_by_key(|it| {
        let ke = (it.r & even_rows).count_ones();
        std::cmp::Reverse(ke.min(g.ne - ke))
    });
    items
}

/// Which work items a run covers: the index range [from, to) of the list above, thinned to the
/// indices ≡ phase (mod stride). Phases of one stride partition the list, so their partial counts
/// add up to the full count; every phase samples the heavy and light items alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection { pub from: usize, pub to: usize, pub stride: usize, pub phase: usize }

impl Selection {
    pub const ALL: Selection = Selection { from: 0, to: usize::MAX, stride: 1, phase: 0 };
    /// Read NQ_PAIR_FROM, NQ_PAIR_TO, NQ_PAIR_STRIDE and NQ_PAIR_PHASE from the environment.
    pub fn from_env() -> Result<Selection, String> {
        let get = |k: &str, d: usize| std::env::var(k).ok().map(|v| v.parse::<usize>().map_err(|_| format!("{k}={v} is not a number"))).unwrap_or(Ok(d));
        let s = Selection { from: get("NQ_PAIR_FROM", 0)?, to: get("NQ_PAIR_TO", usize::MAX)?, stride: get("NQ_PAIR_STRIDE", 1)?.max(1), phase: get("NQ_PAIR_PHASE", 0)? };
        if s.phase >= s.stride { return Err(format!("NQ_PAIR_PHASE {} must be below NQ_PAIR_STRIDE {}", s.phase, s.stride)); }
        Ok(s)
    }
    pub fn is_all(&self, total: usize) -> bool { self.from == 0 && self.to >= total && self.stride == 1 }
    pub fn apply(&self, items: Vec<WorkItem>) -> Vec<WorkItem> {
        items.into_iter().enumerate()
            .filter(|(i, _)| *i >= self.from && *i < self.to && *i % self.stride == self.phase)
            .map(|(_, it)| it).collect()
    }
}

/// Enumerate one partial configuration on the CPU into `table` (dense, `spec.size()` entries, zeroed by the caller):
/// depth-first over the rows of the partial configuration, one queen per row on the columns of its parity, with the usual
/// column / diagonal masks; the diagonal masks are shifted by the gap to the next row of the configuration.
/// Returns (placements tried, partial configurations found).
pub fn enumerate(g: &Geometry, spec: &PartSpec, table: &mut [u64]) -> (u64, u64) {
    let full = g.full;
    let even_cols: u32 = full & 0x5555_5555;
    let odd_cols: u32 = full & 0xAAAA_AAAA;
    let l = spec.rows.len();
    if l == 0 { table[0] += 1; return (0, 1); }
    debug_assert_eq!(table.len(), spec.size());
    let width_odd = spec.width_odd;
    let mut pmask = [0u32; 32];
    let mut gap = [0u32; 32];
    for t in 0..l {
        let r = spec.rows[t];
        pmask[t] = if spec.col_parity(r) == 0 { even_cols } else { odd_cols };
        gap[t] = if t + 1 < l { spec.rows[t + 1] - spec.rows[t] } else { 0 };
    }
    let mut placements = 0u64;
    let mut partials = 0u64;
    let mut st_free = [0u32; 33];
    let mut st_cols = [0u32; 33];
    let mut st_ld = [0u32; 33];
    let mut st_rd = [0u32; 33];
    st_free[0] = pmask[0];
    let last = l - 1;
    let rank_tab = &g.rank_tab[..];
    let mut t = 0usize;
    loop {
        let free = st_free[t];
        if free == 0 {
            if t == 0 { break; }
            t -= 1;
            continue;
        }
        let b = free & free.wrapping_neg();
        st_free[t] = free ^ b;
        placements += 1;
        let cols = st_cols[t] | b;
        if t == last {
            let re = rank_tab[even_bits(cols) as usize] as u64;
            let ro = rank_tab[odd_bits(cols) as usize] as u64;
            table[(re * width_odd + ro) as usize] += 1;
            partials += 1;
            continue;
        }
        let sh = gap[t];
        let ld = ((st_ld[t] | b) << sh) & full;
        let rd = (st_rd[t] | b) >> sh;
        let nt = t + 1;
        st_cols[nt] = cols;
        st_ld[nt] = ld;
        st_rd[nt] = rd;
        st_free[nt] = pmask[nt] & !(cols | ld | rd);
        t = nt;
    }
    (placements, partials)
}

/// Σ_C ta[C] · tb[C̄] for two partial configurations with complementary column structure (b uses the columns a does
/// not). Returns (sum, largest entry of ta).
pub fn join(g: &Geometry, a: &PartSpec, ta: &[u64], b: &PartSpec, tb: &[u64]) -> (u128, u64) {
    assert_eq!(b.ke, g.ne - a.ke);
    assert_eq!(b.ko, g.no - a.ko);
    let ce = g.complement_ranks(0, a.ke);
    let co = g.complement_ranks(1, a.ko);
    let (wa, wb) = (a.width_odd as usize, b.width_odd as usize);
    let mut acc = 0u128;
    let mut maxc = 0u64;
    for ie in 0..a.width_even as usize {
        let row = &ta[ie * wa..(ie + 1) * wa];
        let base = ce[ie] as usize * wb;
        for (io, &e) in row.iter().enumerate() {
            if e == 0 { continue; }
            if e > maxc { maxc = e; }
            acc += e as u128 * tb[base + co[io] as usize] as u128;
        }
    }
    (acc, maxc)
}

pub struct CpuStats {
    /// the count (partial when the selection is not the whole list)
    pub q: u128,
    /// work items in the whole list and in the selection
    pub items_total: usize,
    pub items: usize,
    pub partials: u64,
    pub placements: u64,
    /// largest table entry seen (the GPU stores entries as u32)
    pub max_count: u64,
}

/// The count on the CPU: `threads` workers pull work items in list order, enumerate both partial configurations of
/// each item into dense tables, and join them.
pub fn count_cpu(n: u32, threads: usize, nosym: bool, sel: &Selection) -> CpuStats {
    let g = Geometry::new(n);
    let odd_n = n % 2 == 1;
    let all = work_items(&g, nosym);
    let items_total = all.len();
    let items = sel.apply(all);
    let nitems = items.len();
    let maxw = {
        let k = g.ne / 2;
        (binom(g.ne, k) * binom(g.no, k.min(g.no))) as usize
    };
    let next = AtomicUsize::new(0);
    let res: Vec<(u128, u64, u64, u64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads.max(1)).map(|_| sc.spawn(|| {
            let mut ta = vec![0u64; maxw];
            let mut tb = vec![0u64; maxw];
            let (mut q, mut placements, mut partials, mut maxcount) = (0u128, 0u64, 0u64, 0u64);
            loop {
                let it = next.fetch_add(1, Ordering::Relaxed);
                if it >= nitems { break; }
                let WorkItem { r, partner, weight } = items[it];
                let a = PartSpec::new(&g, r, true);
                ta[..a.size()].iter_mut().for_each(|x| *x = 0);
                let (pl, pc) = enumerate(&g, &a, &mut ta[..a.size()]);
                placements += pl;
                partials += pc;
                let b = PartSpec::new(&g, partner, !odd_n);
                let same = r == partner && !odd_n;
                let (acc, mc) = if same {
                    join(&g, &a, &ta[..a.size()], &a, &ta[..a.size()])
                } else {
                    tb[..b.size()].iter_mut().for_each(|x| *x = 0);
                    let (pl2, pc2) = enumerate(&g, &b, &mut tb[..b.size()]);
                    placements += pl2;
                    partials += pc2;
                    join(&g, &a, &ta[..a.size()], &b, &tb[..b.size()])
                };
                if mc > maxcount { maxcount = mc; }
                q += acc * weight as u128;
            }
            (q, placements, partials, maxcount)
        })).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    CpuStats {
        q: res.iter().map(|r| r.0).sum(),
        items_total,
        items: nitems,
        partials: res.iter().map(|r| r.2).sum(),
        placements: res.iter().map(|r| r.1).sum(),
        max_count: res.iter().map(|r| r.3).max().unwrap_or(0),
    }
}
