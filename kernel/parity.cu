/*
 * Balanced checkerboard join on the GPU: piece enumeration into dense tables, and the join.
 *
 * Every partition (one piece: a row set and a colour) is enumerated by a depth-first search over its
 * rows. The first NQ_PREF rows are decided by a "ticket": a mixed-radix number whose digits choose
 * one column of the right colour in each prefix row (dead prefixes are simply skipped), so the search
 * is split into millions of independent roots that persistent threads pull from a global cursor.
 * Below the prefix the search runs from a lane-strided shared-memory stack of uint4 frames
 * (candidate columns, cols, ld, rd): one pass extracts the lowest candidate of the top frame, writes
 * the shrunken mask back, folds the pop, and pushes the child frame. The frame's row index rides in
 * bits 28-31 of cols (low nibble) and of ld (high nibble); the partition's row mask, colour and last
 * row stay in registers. Leaves are pieces: mode 0 (tabulate) increments the table entry of the
 * piece's (even-column subset rank, odd-column subset rank); mode 1 (read) adds the partner table's
 * entry at the complementary ranks to a per-thread accumulator that is flushed into the pair's sum.
 *
 * Baked by the host: NQ_N, NQ_NE, NQ_NO (columns per colour), NQ_PREF, WG_SIZE (block size),
 * NQ_MAXSTACK (frames per lane; the host checks it against the tallest partition), NQ_T32 (u32
 * table entries, else u64), NQ_RANK2 (two-level colex ranks from shared memory, else a flat table).
 * n <= 28: column masks use bits 0-27 and the row index the nibble above.
 */
typedef unsigned int u32;
typedef unsigned long long u64;
#ifdef NQ_T32
typedef unsigned int tab_t;
#else
typedef unsigned long long tab_t;
#endif

#define FULL ((1u << NQ_N) - 1u)
#define EVEN_COLS (FULL & 0x55555555u)
#define ODD_COLS (FULL & 0xAAAAAAAAu)
#if NQ_N > 28
#error "the row index rides in bits 28-31; requires n <= 28"
#endif

/* compress the even-position bits of x into the low bits (odd_bits: the odd-position bits) */
__device__ __forceinline__ u32 even_bits(u32 x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    x = (x | (x >> 8)) & 0x0000FFFFu;
    return x;
}
__device__ __forceinline__ u32 odd_bits(u32 x) { return even_bits(x >> 1); }

/* leaf: tabulate (md == 0) or read the partner entry at the complementary ranks (md == 1) */
#define LEAF_OP(re, ro) do { \
    if (md) acc += (u64)__ldg(tables + toff + (u64)__ldg(ce + (re)) * wo + __ldg(co + (ro))); \
    else atomicAdd(tables + toff + (u64)(re) * wo + (ro), (tab_t)1); } while (0)

#ifdef NQ_RANK2
/* two-level colex rank: lo[m & 127] + hi[m >> 7][popcount(m & 127)] (rank_tab holds lo[128] then hi[128][8]) */
#define RANK(tab, m) ((tab)[(m) & 127u] + (tab)[128u + (((m) >> 7) << 3) + __popc((m) & 127u)])
#else
#define RANK(tab, m) __ldg(rank_tab + (m))
#endif

/*
 * desc8 (40 bytes per partition): rows[32], nrows, black, mode, pad; desc32 (4 words per partition):
 * table width_odd, offsets of the complement-rank tables (even, odd), pair index (read mode).
 * table_off: table offset in entries; prefix_base: cumulative ticket counts (nparts + 1 entries).
 * tables: the batch's tables; rank_tab / comp: rank and complement-rank tables; pair_sums: per-pair
 * accumulators (read mode); cursor: the global ticket counter; stats[0]: placements.
 */
extern "C" __global__
__launch_bounds__(WG_SIZE)
void parity_dfs(
    const unsigned char* __restrict__ desc8, const u32* __restrict__ desc32,
    const u64* __restrict__ table_off, const u64* __restrict__ prefix_base, u32 nparts,
    tab_t* __restrict__ tables, const u32* __restrict__ rank_tab, const u32* __restrict__ comp,
    u64* __restrict__ pair_sums, u64* __restrict__ cursor, u64* __restrict__ stats)
{
    __shared__ uint4 stack_l[WG_SIZE * NQ_MAXSTACK];
    const u32 lid = threadIdx.x;
#define STK(i) stack_l[(i) * WG_SIZE + lid]
#ifdef NQ_RANK2
    __shared__ u32 rk_s[128 + 128 * 8];
    for (u32 i = lid; i < 128 + 128 * 8; i += WG_SIZE) rk_s[i] = rank_tab[i];
    __syncthreads();
#define RK(m) RANK(rk_s, m)
#else
#define RK(m) RANK(rank_tab, m)
#endif
    const u64 total = prefix_base[nparts];
    u64 placements = 0, acc = 0;
    u32 cur_pair = 0xFFFFFFFFu;
    /* the lane's current partition: row mask, colour, last row, mode, table width and offset */
    u32 rmask = 0, isblack = 1, lastrow = 0, md = 0, wo = 0; u64 toff = 0;
#define PMASK(r) ((((r) & 1u) ^ (isblack ^ 1u)) ? ODD_COLS : EVEN_COLS)
#define NEXTROW(r) ((r) + 1u + __ffs(rmask >> ((r) + 1u)) - 1u)
    const u32* ce = comp; const u32* co = comp;
    u32 sp = 0;
    for (;;) {
        uint4 f;
        if (sp == 0u) {
            /* pull tickets until one yields a live root frame */
            for (;;) {
                const u64 t = atomicAdd(cursor, 1ull);
                if (t >= total) goto done;
                u32 lo = 0, hi = nparts;
                while (hi - lo > 1) { u32 mid = (lo + hi) >> 1; if (prefix_base[mid] <= t) lo = mid; else hi = mid; }
                const u32 b = lo;
                u64 local = t - prefix_base[b];
                const unsigned char* rw = desc8 + 40u * b;
                const u32 L = rw[32]; isblack = rw[33]; md = rw[34];
                toff = table_off[b]; wo = desc32[4u * b];
                ce = comp + desc32[4u * b + 1]; co = comp + desc32[4u * b + 2];
                if (md) { u32 pr = desc32[4u * b + 3]; if (pr != cur_pair) { if (cur_pair != 0xFFFFFFFFu && acc) atomicAdd(pair_sums + cur_pair, acc); acc = 0; cur_pair = pr; } }
                rmask = 0; for (u32 i = 0; i < L; ++i) rmask |= 1u << rw[i];
                lastrow = rw[L - 1];
                const u32 d = (NQ_PREF < L) ? NQ_PREF : L;
                u32 cols = 0, ld = 0, rd = 0; bool ok = true;
                for (u32 i = 0; i < d; ++i) {
                    const u32 r = rw[i];
                    const u32 cp = (PMASK(r) == ODD_COLS) ? 1u : 0u;
                    const u32 radix = cp ? NQ_NO : NQ_NE;
                    const u32 q = (u32)(local % radix); local /= radix;
                    const u32 bit = 1u << (2u * q + cp);
                    if ((cols | ld | rd) & bit) { ok = false; break; }
                    placements++;
                    const u32 gp = (i + 1 < L) ? (u32)(rw[i + 1] - r) : 0u;
                    cols |= bit; ld = ((ld | bit) << gp) & FULL; rd = (rd | bit) >> gp;
                }
                if (!ok) continue;
                if (d == L) {   /* the prefix is the whole piece */
                    u32 re = RK(even_bits(cols)); u32 ro = RK(odd_bits(cols));
                    LEAF_OP(re, ro);
                    continue;
                }
                const u32 r0 = rw[d];
                const u32 cand = PMASK(r0) & ~(cols | ld | rd);
                if (cand == 0u) continue;
                f = make_uint4(cand, cols | ((r0 & 15u) << 28), ld | ((r0 >> 4) << 28), rd);
                STK(0) = f; sp = 1u;
                break;
            }
        } else {
            f = STK(sp - 1u);
        }
        /* one pass: extract the lowest candidate of the top frame, fold the pop, push the child */
        const u32 r = (f.y >> 28) | ((f.z >> 28) << 4);   /* row of this frame */
        const u32 m = f.x;
        const u32 rext = m & (0u - m);
        const u32 remaining = m ^ rext;
        STK(sp - 1u).x = remaining;
        sp -= (remaining == 0u) ? 1u : 0u;
        placements++;
        const u32 cols = (f.y & FULL) | rext;
        if (r == lastrow) {
            u32 re = RK(even_bits(cols)); u32 ro = RK(odd_bits(cols));
            LEAF_OP(re, ro);
            continue;
        }
        const u32 r1 = NEXTROW(r);
        const u32 sh = r1 - r;
        const u32 ld = (((f.z & FULL) | rext) << sh) & FULL;
        const u32 rd = ((f.w & FULL) | rext) >> sh;
        const u32 cand = PMASK(r1) & ~(cols | ld | rd);
        STK(sp) = make_uint4(cand, cols | ((r1 & 15u) << 28), ld | ((r1 >> 4) << 28), rd);
        sp += (cand != 0u) ? 1u : 0u;
    }
done:
    if (cur_pair != 0xFFFFFFFFu && acc) atomicAdd(pair_sums + cur_pair, acc);
    atomicAdd(stats, placements);
}

/* Join of a table with itself (pairs whose partner is the same row set): one block per pair,
 * Σ_i ta[i] · tb[complement of i], reduced in shared memory. */
extern "C" __global__ void parity_join(
    const tab_t* __restrict__ tables, const u64* __restrict__ off_a, const u64* __restrict__ off_b,
    const u32* __restrict__ we_a, const u32* __restrict__ wo_a, const u32* __restrict__ wo_b,
    const u32* __restrict__ ce_off, const u32* __restrict__ co_off, const u32* __restrict__ comp,
    u64* __restrict__ out)
{
    const u32 p = blockIdx.x;
    const tab_t* ta = tables + off_a[p]; const tab_t* tb = tables + off_b[p];
    const u32 we = we_a[p], wa = wo_a[p], wb = wo_b[p];
    const u32* ce = comp + ce_off[p]; const u32* co = comp + co_off[p];
    const u64 n = (u64)we * wa;
    u64 acc = 0;
    for (u64 i = threadIdx.x; i < n; i += blockDim.x) {
        u64 e = (u64)ta[i];
        if (e == 0) continue;
        u32 ie = (u32)(i / wa), io = (u32)(i % wa);
        acc += e * (u64)tb[(u64)ce[ie] * wb + co[io]];
    }
    __shared__ u64 red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (u32 s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[p] = red[0];
}

/* Running maximum of the batch's table entries (NQ_TABMAX=1): the guard for the u32 entry type. */
extern "C" __global__ void parity_tabmax(const tab_t* __restrict__ tables, u64 count, u64* __restrict__ out)
{
    u64 m = 0;
    for (u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (u64)gridDim.x * blockDim.x) { u64 v = (u64)tables[i]; if (v > m) m = v; }
    __shared__ u64 red[256];
    red[threadIdx.x] = m; __syncthreads();
    for (u32 s = blockDim.x >> 1; s > 0; s >>= 1) { if (threadIdx.x < s && red[threadIdx.x + s] > red[threadIdx.x]) red[threadIdx.x] = red[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) atomicMax(out, red[0]);
}
