//! Nearest neighbour algorithms from `ann-search-rs`

pub mod nearest_neighbour_cpu;
#[cfg(feature = "gpu")]
pub mod nearest_neighbour_gpu;
