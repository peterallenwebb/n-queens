//! Counting n-queens solutions by the balanced checkerboard join.
//!
//! Call a cell (r, c) even or odd by the parity of r + c (the checkerboard colouring). Every diagonal
//! has constant parity, so the queens on even cells and the queens on odd cells can only attack each
//! other through rows and columns. A configuration (a solution) therefore splits into an even
//! configuration and an odd configuration — partial solutions confined to one parity — with
//! complementary row sets and complementary column sets, and the count factors into a sum of products
//! of partial-configuration counts (METHOD.md). This crate contains exactly the code used to compute Q(n) that way:
//!
//! * [`geometry`] — bit tricks and the colex ranking of column subsets (both host and device use it);
//! * [`parity`] — work items (row-set pairs), the CPU enumerator, the join, and the CPU count
//!   (the validation path, exact for n ≤ 22 in hours on a laptop);
//! * [`gpu`] — the CUDA production path (feature `cuda`): the same decomposition with the configuration
//!   enumeration and the join on the device;
//! * [`reference`] — an independent brute-force solver used only by the tests;
//! * [`burnside`] — the mod-8 consistency check every full count must pass;
//! * [`known`] — the published values of Q(n), n ≤ 27 (OEIS A000170), for the tests and the ladder.
#![allow(clippy::manual_isolate_lowest_one, clippy::needless_range_loop)]

pub mod burnside;
pub mod geometry;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod known;
pub mod parity;
pub mod reference;
