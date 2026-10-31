//! Re-exports of commonly used types, traits, structures and functions across
//! the crate:
//!
//! ```rust
//! use evoc_rs::prelude::*;
//! ```

pub use crate::errors::EvocErrors;
pub use crate::graph::EvocEmbeddingParams;
pub use crate::nearest_neighbours::nearest_neighbour_cpu::NearestNeighbourParamsEvoc;
pub use crate::utils::input::{EvocMatrix, MatInput};
pub use crate::utils::sparse::CoordinateList;
pub use crate::utils::traits::EvocFloat;
pub use ann_search_rs::prelude::{Verbosity, parse_annsearch_verbosity as parse_verbosity_level};

#[cfg(feature = "gpu")]
pub use crate::nearest_neighbours::nearest_neighbour_gpu::NearestNeighbourParamsGpuEvoc;

///////////
// Types //
///////////

/// The kNN search results in manifolds. If Ok, it's (indices, distances);
/// otherwise a [EvocErrors].
///
/// ### Fields
///
/// If successful:
///
/// * `0` - The indices of the nearest neighbours excluding self.
/// * `1` - The distances of the nearest neighbours excluding self.
pub type EvocKnnResults<T> = Result<(Vec<Vec<usize>>, Vec<Vec<T>>), EvocErrors>;

/// Type for the pre-computed kNN
///
/// ### Fields
///
/// * `0` - The indices of the nearest neighbours excluding self.
/// * `1` - The distances of the nearest neighbours excluding self.
pub type PreComputedKnn<T> = Option<(Vec<Vec<usize>>, Vec<Vec<T>>)>;
