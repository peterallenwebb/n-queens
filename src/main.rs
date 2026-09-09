//! Command line: count Q(n) by the balanced checkerboard join on the CPU or on a CUDA GPU, run the
//! brute-force reference, or run the mod-8 check on a claimed count. RUNBOOK.md documents the
//! environment variables that select work-item ranges and phases and tune the GPU.
use nqueens_parity::burnside::verify_mod8;
use nqueens_parity::known::known;
use nqueens_parity::parity::{count_cpu, work_items, Selection};
use nqueens_parity::{geometry::Geometry, reference};
use std::process::exit;
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage: nqueens-parity <N> --cpu [--threads T] [--nosym]      count on the CPU (validation; n <= 22 is hours)\n\
       \x20      nqueens-parity <N> --cuda [--device i] [--nosym]      count on a CUDA GPU (build with --features cuda)\n\
       \x20      nqueens-parity <N> --reference                        brute-force count (n <= 16)\n\
       \x20      nqueens-parity <N> --mod8 <count>                     mod-8 (Burnside) check of a claimed count\n\
       \x20      nqueens-parity <N> --items                            number of work items (for choosing a stride)\n\
       env: NQ_PAIR_FROM/NQ_PAIR_TO (work-item range), NQ_PAIR_STRIDE/NQ_PAIR_PHASE (phase p of S), and for --cuda\n\
       \x20    NQ_PARITY_MEM NQ_PREF NQ_THREADS_GPU NQ_MAXSTACK NQ_SLM NQ_RANK2 NQ_T32 NQ_TABMAX NQ_MAXBATCH NQ_VERBOSE (RUNBOOK.md)"
    );
    exit(2);
}

fn parse<T: std::str::FromStr>(args: &mut std::env::Args, flag: &str) -> T {
    match args.next().and_then(|v| v.parse().ok()) { Some(v) => v, None => { eprintln!("error: {flag} needs a value"); usage() } }
}

fn check_mod8(n: u32, q: u128) -> bool {
    let t = Instant::now();
    let (s2, pass) = verify_mod8(n, q);
    if pass {
        println!("mod-8 check: PASS (Q ≡ {} mod 8, {s2} C2-symmetric solutions, {:.2}s)", q % 8, t.elapsed().as_secs_f64());
    } else {
        println!("MOD-8 CHECK FAILED: Q = {q} ≡ {} (mod 8) but C2-symmetric count {s2} ≡ {} (mod 8) — one of them is wrong", q % 8, s2 % 8);
    }
    pass
}

fn main() {
    let mut args = std::env::args();
    args.next();
    let (mut n, mut cpu, mut cuda, mut reference, mut items, mut nosym) = (None::<u32>, false, false, false, false, false);
    let mut threads: usize = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let mut device: usize = 0;
    let mut mod8: Option<u128> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--cpu" => cpu = true,
            "--cuda" => cuda = true,
            "--reference" => reference = true,
            "--items" => items = true,
            "--nosym" => nosym = true,
            "--threads" => threads = parse(&mut args, "--threads"),
            "--device" => device = parse(&mut args, "--device"),
            "--mod8" => mod8 = Some(parse(&mut args, "--mod8")),
            _ => match a.parse::<u32>() { Ok(v) if n.is_none() => n = Some(v), _ => usage() },
        }
    }
    let Some(n) = n else { usage() };
    let _ = &device;   // only the cuda path reads it
    if !(1..=28).contains(&n) { eprintln!("error: N must be in 1..=28"); exit(2); }
    if let Some(q) = mod8 {
        exit(if check_mod8(n, q) { 0 } else { 3 });
    }
    if items {
        let g = Geometry::new(n);
        println!("N={n}: {} work items ({} without the 180° symmetry)", work_items(&g, false).len(), work_items(&g, true).len());
        return;
    }
    if reference {
        if n > 18 { eprintln!("error: --reference is brute force; keep N <= 18"); exit(2); }
        let t = Instant::now();
        let q = reference::count(n);
        println!("N={n}: {q} solutions (brute force, {:.2}s) {}", t.elapsed().as_secs_f64(), match known(n) { Some(k) if k == q as u128 => "ok", Some(_) => "MISMATCH", None => "" });
        return;
    }
    let sel = match Selection::from_env() { Ok(s) => s, Err(e) => { eprintln!("error: {e}"); exit(2) } };
    if cpu {
        let t = Instant::now();
        let st = count_cpu(n, threads, nosym, &sel);
        let partial = !sel.is_all(st.items_total);
        if partial { println!("chunk: work items [{}, {}) of {}, stride {} phase {}: {} items — partial count", sel.from, sel.to.min(st.items_total), st.items_total, sel.stride, sel.phase, st.items); }
        let wall = t.elapsed().as_secs_f64();
        let status = if partial { "PARTIAL" } else { match known(n) { Some(k) if k == st.q => "ok", Some(_) => "MISMATCH", None => "unknown" } };
        println!("parity join: Q={} {status} | placements {} ({:.3}/sol, {:.3e}/s) | {} pairs | pieces {} | max table entry {} | {threads} threads | {wall:.1}s",
            st.q, st.placements, st.placements as f64 / st.q.max(1) as f64, st.placements as f64 / wall, st.items, st.pieces, st.max_count);
        if partial { println!("(partial run: no mod-8 check)"); exit(0); }
        let pass = check_mod8(n, st.q);
        exit(if status == "MISMATCH" || !pass { 3 } else { 0 });
    }
    if cuda {
        #[cfg(feature = "cuda")]
        {
            use nqueens_parity::gpu;
            let cfg = match gpu::GpuConfig::from_env(n) { Ok(c) => c, Err(e) => { eprintln!("error: {e}"); exit(2) } };
            let st = match gpu::run(n, device, nosym, &sel, &cfg) { Ok(s) => s, Err(e) => { eprintln!("{e}"); exit(1) } };
            let status = gpu::status(n, st.q, st.partial);
            println!("parity join: Q={} {status} | placements {} ({:.3}/sol, {:.3e}/s) | {} pairs | {} launches | {} batches | {:.1}s",
                st.q, st.placements, st.placements as f64 / st.q.max(1) as f64, st.placements as f64 / st.wall_s, st.items, st.launches, st.batches, st.wall_s);
            if st.partial { println!("(partial run: no mod-8 check)"); exit(0); }
            let pass = check_mod8(n, st.q);
            exit(if status == "MISMATCH" || !pass { 3 } else { 0 });
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("error: this binary was built without the cuda feature (cargo build --release --features cuda)");
            exit(2);
        }
    }
    usage();
}
