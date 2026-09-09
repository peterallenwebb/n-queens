//! The CUDA production path (feature `cuda`): the balanced checkerboard join with the configuration
//! enumeration and the join on the device. Same work items, tables and join as the CPU path
//! (`parity`); the kernel is `kernel/parity.cu`, compiled at run time by NVRTC for the device found.
//!
//! Work items are processed in batches: a batch takes as many items as fit its table budget, one
//! dense table per item for the side with the smaller table. Phase 1 enumerates those configurations into
//! their tables; phase 2 enumerates the partner configurations and, at each leaf, reads the partner's table
//! entry at the complementary column ranks into a per-pair sum (so the larger side is never stored).
//! Items whose partner is the same row set (even n, R = ρ(R̄)) are joined by a separate kernel over
//! the table with itself. Sums are multiplied by the item weights and accumulated in u128 on the host.
use crate::geometry::Geometry;
use crate::known::known;
use crate::parity::{work_items, PartSpec, Selection, WorkItem};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::time::Instant;

const KERNEL_SRC: &str = include_str!("../kernel/parity.cu");

fn err(e: impl std::fmt::Display) -> String { format!("CUDA error: {e}") }

/// Run configuration, read from the environment by `GpuConfig::from_env` (RUNBOOK.md lists the
/// variables). The defaults are the measured best configuration on an RTX 5090.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// device memory for the batch tables, bytes (NQ_PARITY_MEM, default 8 GiB)
    pub table_budget: usize,
    /// prefix rows decided by the ticket (NQ_PREF; default 3, +1 at n ≥ 24, +1 at n ≥ 28)
    pub pref: u32,
    /// persistent threads (NQ_THREADS_GPU, default 139,264)
    pub threads: usize,
    /// frames per lane in the shared-memory stack (NQ_MAXSTACK, default n − pref + 1; checked)
    pub maxstack: u32,
    /// block size (NQ_SLM; default 128 when the static shared memory fits, else 64)
    pub block: usize,
    /// two-level shared-memory ranks (NQ_RANK2, default 1)
    pub rank2: bool,
    /// u32 table entries (NQ_T32, default 1; 0 = u64)
    pub t32: bool,
    /// report the largest table entry of the run (NQ_TABMAX=1)
    pub tabmax: bool,
    /// stop after this many batches — a partial run for probing (NQ_MAXBATCH)
    pub maxbatch: usize,
    /// per-batch progress lines (NQ_VERBOSE=1)
    pub verbose: bool,
    /// NVRTC target architecture override (NQ_NVRTC_ARCH, e.g. sm_120)
    pub nvrtc_arch: Option<String>,
}

impl GpuConfig {
    pub fn from_env(n: u32) -> Result<GpuConfig, String> {
        let num = |k: &str| -> Result<Option<usize>, String> {
            match std::env::var(k) { Ok(v) => v.parse::<usize>().map(Some).map_err(|_| format!("{k}={v} is not a number")), Err(_) => Ok(None) }
        };
        let flag = |k: &str, d: bool| std::env::var(k).map(|v| v == "1").unwrap_or(d);
        let pref = num("NQ_PREF")?.map(|v| v as u32).unwrap_or(3 + (n >= 24) as u32 + (n >= 28) as u32);
        if pref == 0 || pref > n { return Err(format!("NQ_PREF {pref} must be in 1..={n}")); }
        let maxstack = num("NQ_MAXSTACK")?.map(|v| v as u32).unwrap_or(n - pref + 1);
        let rank2 = flag("NQ_RANK2", true);
        let smem_extra: usize = if rank2 { (128 + 128 * 8) * 4 } else { 0 };
        let block_default = if 128 * 16 * maxstack as usize + smem_extra <= 48 * 1024 { 128 } else { 64 };
        let block = num("NQ_SLM")?.unwrap_or(block_default);
        if block == 0 || block > 1024 || !block.is_power_of_two() { return Err(format!("NQ_SLM {block} must be a power of two up to 1024")); }
        Ok(GpuConfig {
            table_budget: num("NQ_PARITY_MEM")?.unwrap_or(8usize << 30),
            pref, threads: num("NQ_THREADS_GPU")?.unwrap_or(139_264), maxstack, block, rank2,
            t32: flag("NQ_T32", true), tabmax: flag("NQ_TABMAX", false),
            maxbatch: num("NQ_MAXBATCH")?.unwrap_or(usize::MAX), verbose: flag("NQ_VERBOSE", false),
            nvrtc_arch: std::env::var("NQ_NVRTC_ARCH").ok(),
        })
    }
}

pub struct GpuStats {
    /// the count (partial when `partial`)
    pub q: u128,
    pub items_total: usize,
    pub items: usize,
    pub placements: u64,
    pub launches: u64,
    pub batches: usize,
    /// true when the selection was not the whole list or NQ_MAXBATCH stopped the run
    pub partial: bool,
    /// largest table entry, when NQ_TABMAX=1
    pub table_max: Option<u64>,
    pub wall_s: f64,
}

/// Count Q(n) (or the selected part of it) on CUDA device `device_index`.
pub fn run(n: u32, device_index: usize, nosym: bool, sel: &Selection, cfg: &GpuConfig) -> Result<GpuStats, String> {
    let t0 = Instant::now();
    let g = Geometry::new(n);
    let odd_n = n % 2 == 1;
    let all: Vec<WorkItem> = work_items(&g, nosym);
    let items_total = all.len();
    let chunked = !sel.is_all(items_total);
    let items: Vec<WorkItem> = sel.apply(all);
    if chunked {
        println!("chunk: work items [{}, {}) of {items_total}, stride {} phase {}: {} items — partial count",
            sel.from, sel.to.min(items_total), sel.stride, sel.phase, items.len());
    }
    let dev = CudaDevice::new(device_index).map_err(err)?;
    println!("device: CUDA {device_index} ({})", dev.name().map_err(err)?);
    println!("N={n} PARITY JOIN: {} work items (sym {}), prefix depth {}, table budget {} MiB, {} persistent threads, read-side join",
        items.len(), if nosym { "off" } else { "180°" }, cfg.pref, cfg.table_budget >> 20, cfg.threads);
    println!("machine: stack-resident (WG_SIZE {}, {} frames = {} B smem/block{})", cfg.block, cfg.maxstack,
        cfg.block * 16 * cfg.maxstack as usize, if cfg.rank2 { " + 4.6 KB rank tables" } else { "" });
    // Stack-height guard: a partition with L rows needs L - min(pref, L) frames (one per row after the
    // prefix). The row count varies per work item (any even count for even n, any odd count for odd
    // n), so every item is checked; an undersized stack would overrun the lane stacks silently.
    let frames_needed = |rm: u32| { let l = rm.count_ones(); l - l.min(cfg.pref) };
    let (stack_need, tallest) = items.iter().map(|it| {
        let a = frames_needed(it.r); let b = frames_needed(it.partner);
        if a >= b { (a, it.r.count_ones()) } else { (b, it.partner.count_ones()) }
    }).max().unwrap_or((0, 0));
    if stack_need > cfg.maxstack {
        return Err(format!("NQ_MAXSTACK {} frames < {stack_need} needed by a {tallest}-row partition (n {n}, prefix {})", cfg.maxstack, cfg.pref));
    }
    println!("stack: tallest partition {tallest} rows needs {stack_need} frames of {}", cfg.maxstack);
    if cfg.rank2 && (g.ne > 14 || g.no > 14) { return Err("NQ_RANK2 two-level ranks cover at most 14 columns per parity (n <= 28)".into()); }
    if cfg.t32 { println!("tables: u32 entries"); } else { println!("tables: u64 entries"); }
    if cfg.rank2 { println!("rank: two-level shared-memory tables"); } else { println!("rank: flat table"); }

    let src = format!("#define NQ_N {n}u\n#define NQ_NE {}u\n#define NQ_NO {}u\n#define NQ_PREF {}u\n#define WG_SIZE {}\n#define NQ_MAXSTACK {}\n{}{}{}",
        g.ne, g.no, cfg.pref, cfg.block, cfg.maxstack,
        if cfg.t32 { "#define NQ_T32 1\n" } else { "" }, if cfg.rank2 { "#define NQ_RANK2 1\n" } else { "" }, KERNEL_SRC);
    let mut copts = cudarc::nvrtc::CompileOptions::default();
    if let Some(a) = &cfg.nvrtc_arch { println!("nvrtc arch: {a}"); copts.arch = Some(Box::leak(a.clone().into_boxed_str())); }
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, copts).map_err(|e| format!("NVRTC build failed:\n{e}"))?;
    dev.load_ptx(ptx, "nqp", &["parity_dfs", "parity_join", "parity_tabmax"]).map_err(err)?;
    let f_dfs = dev.get_func("nqp", "parity_dfs").ok_or("kernel parity_dfs not found")?;
    let f_join = dev.get_func("nqp", "parity_join").ok_or("kernel parity_join not found")?;
    let f_tabmax = dev.get_func("nqp", "parity_tabmax").ok_or("kernel parity_tabmax not found")?;
    let tabmax_d: CudaSlice<u64> = dev.alloc_zeros(1).map_err(err)?;

    // complement-rank tables, concatenated: comp_base[which][k] is the offset of the k-subset table
    let mut comp: Vec<u32> = Vec::new();
    let mut comp_base = vec![vec![0u32; g.ne.max(g.no) as usize + 1]; 2];
    for which in 0..2usize {
        let cnt = if which == 0 { g.ne } else { g.no };
        for k in 0..=cnt { comp_base[which][k as usize] = comp.len() as u32; comp.extend(g.complement_ranks(which, k)); }
    }
    let comp_d: CudaSlice<u32> = dev.htod_sync_copy(&comp).map_err(err)?;
    let rank_host: Vec<u32> = if cfg.rank2 { Geometry::two_level_rank_table() } else { g.rank_tab.clone() };
    let rank_d: CudaSlice<u32> = dev.htod_sync_copy(&rank_host).map_err(err)?;
    let entry_bytes: usize = if cfg.t32 { 4 } else { 8 };
    let batch_entries = cfg.table_budget / entry_bytes;
    let mut tables_d: CudaSlice<u64> = dev.alloc_zeros(cfg.table_budget / 8).map_err(err)?;
    let mut cursor_d: CudaSlice<u64> = dev.alloc_zeros(1).map_err(err)?;
    let stats_d: CudaSlice<u64> = dev.alloc_zeros(2).map_err(err)?;

    // tickets of a partition: product over its prefix rows of the number of columns of the row's parity
    let prefix_count = |spec: &PartSpec| -> u64 {
        let d = (cfg.pref as usize).min(spec.rows.len());
        spec.rows[..d].iter().map(|&r| if spec.col_parity(r) == 0 { g.ne as u64 } else { g.no as u64 }).product()
    };
    // per-partition descriptors of one launch (see the kernel header for the layout)
    struct Launch { desc8: Vec<u8>, desc32: Vec<u32>, off: Vec<u64>, pbase: Vec<u64>, n: usize }
    impl Launch {
        fn new() -> Self { Launch { desc8: vec![], desc32: vec![], off: vec![], pbase: vec![0], n: 0 } }
        fn push(&mut self, sp: &PartSpec, off: u64, wo: u32, mode: u8, ceo: u32, coo: u32, pair: u32, tickets: u64) {
            let mut r = [0u8; 40];
            for (j, &x) in sp.rows.iter().enumerate() { r[j] = x as u8; }
            r[32] = sp.rows.len() as u8; r[33] = sp.even as u8; r[34] = mode;
            self.desc8.extend_from_slice(&r);
            self.desc32.extend_from_slice(&[wo, ceo, coo, pair]);
            self.off.push(off);
            let last = *self.pbase.last().unwrap();
            self.pbase.push(last + tickets);
            self.n += 1;
        }
    }
    let mut cap: usize = 4096;
    let mut desc8_d: CudaSlice<u8> = dev.alloc_zeros(40 * cap).map_err(err)?;
    let mut desc32_d: CudaSlice<u32> = dev.alloc_zeros(4 * cap).map_err(err)?;
    let mut off_d: CudaSlice<u64> = dev.alloc_zeros(cap).map_err(err)?;
    let mut pbase_d: CudaSlice<u64> = dev.alloc_zeros(cap + 1).map_err(err)?;
    let mut sums_d: CudaSlice<u64> = dev.alloc_zeros(cap).map_err(err)?;
    let mut jo_d: CudaSlice<u64> = dev.alloc_zeros(cap).map_err(err)?;
    let mut jw_d: CudaSlice<u32> = dev.alloc_zeros(5 * cap).map_err(err)?;
    let mut jout_d: CudaSlice<u64> = dev.alloc_zeros(cap).map_err(err)?;

    let mut q: u128 = 0;
    let mut placements: u64 = 0;
    let mut launches: u64 = 0;
    let mut pairs_done: usize = 0;
    let mut last_report = Instant::now();
    let mut idx = 0usize;
    let mut batches = 0usize;
    let mut stopped = false;
    while idx < items.len() {
        if batches >= cfg.maxbatch { println!("NQ_MAXBATCH reached: stopping after {batches} batches (partial count)"); stopped = true; break; }
        batches += 1;
        // ---- assemble a batch: tabulate the side with the smaller table, read from the other
        let mut tab: Vec<(PartSpec, u64)> = Vec::new();
        let mut reads: Vec<(PartSpec, usize, u32)> = Vec::new();
        let mut selfp: Vec<(usize, u32)> = Vec::new();
        let mut weights: Vec<u32> = Vec::new();
        let mut used = 0u64;
        while idx < items.len() {
            let it = &items[idx];
            let a = PartSpec::new(&g, it.r, true);
            let same = it.r == it.partner && !odd_n;
            let b = PartSpec::new(&g, it.partner, !odd_n);
            let (t, r) = if same || a.size() <= b.size() { (a.clone(), b) } else { (b.clone(), a) };
            let need = t.size() as u64;
            if used + need > batch_entries as u64 {
                if tab.is_empty() { return Err(format!("table budget too small for one table ({need} entries)")); }
                break;
            }
            let pair = weights.len() as u32;
            weights.push(it.weight);
            let ti = tab.len();
            tab.push((t, used));
            used += need;
            if same { selfp.push((ti, pair)); } else { reads.push((r, ti, pair)); }
            idx += 1;
        }
        let npairs = weights.len();
        let nparts_max = tab.len().max(reads.len()).max(1);
        if nparts_max > cap || npairs > cap {
            cap = nparts_max.max(npairs).next_power_of_two();
            desc8_d = dev.alloc_zeros(40 * cap).map_err(err)?; desc32_d = dev.alloc_zeros(4 * cap).map_err(err)?;
            off_d = dev.alloc_zeros(cap).map_err(err)?; pbase_d = dev.alloc_zeros(cap + 1).map_err(err)?;
            sums_d = dev.alloc_zeros(cap).map_err(err)?; jo_d = dev.alloc_zeros(cap).map_err(err)?;
            jw_d = dev.alloc_zeros(5 * cap).map_err(err)?; jout_d = dev.alloc_zeros(cap).map_err(err)?;
        }
        {
            let words = ((used as usize) * entry_bytes + 7) / 8;
            let mut used_view = tables_d.slice_mut(0..words);
            dev.memset_zeros(&mut used_view).map_err(err)?;
            let mut sv = sums_d.slice_mut(0..npairs.max(1));
            dev.memset_zeros(&mut sv).map_err(err)?;
        }
        let mut launch = |ph: &Launch| -> Result<(), String> {
            if ph.n == 0 { return Ok(()); }
            { let mut v = desc8_d.slice_mut(0..ph.desc8.len()); dev.htod_sync_copy_into(&ph.desc8, &mut v).map_err(err)?; }
            { let mut v = desc32_d.slice_mut(0..ph.desc32.len()); dev.htod_sync_copy_into(&ph.desc32, &mut v).map_err(err)?; }
            { let mut v = off_d.slice_mut(0..ph.off.len()); dev.htod_sync_copy_into(&ph.off, &mut v).map_err(err)?; }
            { let mut v = pbase_d.slice_mut(0..ph.pbase.len()); dev.htod_sync_copy_into(&ph.pbase, &mut v).map_err(err)?; }
            dev.htod_sync_copy_into(&[0u64], &mut cursor_d).map_err(err)?;
            let lc = LaunchConfig { grid_dim: ((cfg.threads / cfg.block) as u32, 1, 1), block_dim: (cfg.block as u32, 1, 1), shared_mem_bytes: 0 };
            unsafe {
                f_dfs.clone().launch(lc, (&desc8_d, &desc32_d, &off_d, &pbase_d, ph.n as u32, &tables_d, &rank_d, &comp_d, &sums_d, &cursor_d, &stats_d)).map_err(err)?;
            }
            dev.synchronize().map_err(err)?;
            Ok(())
        };
        // ---- phase 1: tabulate
        let mut ph1 = Launch::new();
        for (sp, off) in &tab { ph1.push(sp, *off, sp.width_odd as u32, 0, 0, 0, 0, prefix_count(sp)); }
        let l0 = Instant::now();
        launch(&ph1)?;
        launches += 1;
        let t_tab = l0.elapsed().as_secs_f64();
        if cfg.tabmax {
            let lc = LaunchConfig { grid_dim: (1024, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            unsafe { f_tabmax.clone().launch(lc, (&tables_d, used, &tabmax_d)).map_err(err)?; }
            dev.synchronize().map_err(err)?;
        }
        // ---- phase 2: read-side leaves accumulate into the pair sums
        let mut ph2 = Launch::new();
        for (sp, ti, pair) in &reads {
            let (t, off) = &tab[*ti];
            debug_assert_eq!(t.ke, g.ne - sp.ke);
            debug_assert_eq!(t.ko, g.no - sp.ko);
            ph2.push(sp, *off, t.width_odd as u32, 1, comp_base[0][sp.ke as usize], comp_base[1][sp.ko as usize], *pair, prefix_count(sp));
        }
        let l1 = Instant::now();
        launch(&ph2)?;
        launches += 1;
        let t_read = l1.elapsed().as_secs_f64();
        // ---- self-pairs: join kernel over the table with itself
        if !selfp.is_empty() {
            let ns = selfp.len();
            let mut oa = vec![0u64; ns];
            let mut w = vec![0u32; 5 * ns];
            for (i, (ti, _)) in selfp.iter().enumerate() {
                let (t, off) = &tab[*ti];
                oa[i] = *off; w[i] = t.width_even as u32; w[ns + i] = t.width_odd as u32; w[2 * ns + i] = t.width_odd as u32;
                w[3 * ns + i] = comp_base[0][t.ke as usize]; w[4 * ns + i] = comp_base[1][t.ko as usize];
            }
            { let mut v = jo_d.slice_mut(0..ns); dev.htod_sync_copy_into(&oa, &mut v).map_err(err)?; }
            { let mut v = jw_d.slice_mut(0..5 * ns); dev.htod_sync_copy_into(&w, &mut v).map_err(err)?; }
            let jcfg = LaunchConfig { grid_dim: (ns as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let we_v = jw_d.slice(0..ns); let wa_v = jw_d.slice(ns..2 * ns); let wb_v = jw_d.slice(2 * ns..3 * ns);
            let ceo_v = jw_d.slice(3 * ns..4 * ns); let coo_v = jw_d.slice(4 * ns..5 * ns);
            let jo_v = jo_d.slice(0..ns);
            unsafe { f_join.clone().launch(jcfg, (&tables_d, &jo_v, &jo_v, &we_v, &wa_v, &wb_v, &ceo_v, &coo_v, &comp_d, &jout_d)).map_err(err)?; }
            launches += 1;
            let mut out_h = vec![0u64; ns];
            { let v = jout_d.slice(0..ns); dev.dtoh_sync_copy_into(&v, &mut out_h).map_err(err)?; }
            for (i, (_, pair)) in selfp.iter().enumerate() { q += out_h[i] as u128 * weights[*pair as usize] as u128; }
        }
        let mut sums_h = vec![0u64; npairs.max(1)];
        { let v = sums_d.slice(0..npairs.max(1)); dev.dtoh_sync_copy_into(&v, &mut sums_h).map_err(err)?; }
        for (_, _, pair) in &reads { q += sums_h[*pair as usize] as u128 * weights[*pair as usize] as u128; }
        let mut st = [0u64; 2];
        dev.dtoh_sync_copy_into(&stats_d, &mut st).map_err(err)?;
        placements = st[0];
        pairs_done += npairs;
        if cfg.verbose || last_report.elapsed().as_secs() >= 5 {
            last_report = Instant::now();
            println!("[{:8.1}s] batch: {npairs} pairs ({} tabulated, {} read, {} self), {} MiB tables, tab {t_tab:.3}s read {t_read:.3}s; pairs {pairs_done}/{}; placements {placements} ({:.3e}/s)",
                t0.elapsed().as_secs_f64(), tab.len(), reads.len(), selfp.len(), used as usize * entry_bytes >> 20, items.len(), placements as f64 / t0.elapsed().as_secs_f64());
        }
    }
    let table_max = if cfg.tabmax {
        let mut m = [0u64];
        dev.dtoh_sync_copy_into(&tabmax_d, &mut m).map_err(err)?;
        println!("table entry maximum over the run: {} (u32 limit 4294967295)", m[0]);
        Some(m[0])
    } else { None };
    Ok(GpuStats { q, items_total, items: items.len(), placements, launches, batches, partial: chunked || stopped, table_max, wall_s: t0.elapsed().as_secs_f64() })
}

/// Status word of a run: "ok" / "MISMATCH" against the published value, "unknown" beyond it, or
/// "PARTIAL" for a selection.
pub fn status(n: u32, q: u128, partial: bool) -> &'static str {
    if partial { "PARTIAL" } else {
        match known(n) { Some(k) if k == q => "ok", Some(_) => "MISMATCH", None => "unknown" }
    }
}
