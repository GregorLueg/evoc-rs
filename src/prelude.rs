//! Re-exports of commonly used types, traits, structures and functions across
//! the crate:
//!
//! ```rust
//! use evoc_rs::prelude::*;
//! ```

pub use crate::errors::EvocErrors;
pub use crate::graph::EvocEmbeddingParams;
pub use crate::utils::sparse::CoordinateList;
pub use crate::utils::traits::EvocFloat;

///////////
// Enums //
///////////

/// Enum that controls verbosity
#[derive(Clone, Copy, Debug, Default)]
pub enum Verbosity {
    /// No verbosity at all
    #[default]
    Quiet,
    /// Normal levels of verbosity
    Normal,
    /// Detailed verbosity with increased messages
    Detailed,
}

impl Verbosity {
    /// Returns true if normal or detailed verbosity is set
    pub fn normal_verbosity(&self) -> bool {
        matches!(self, Verbosity::Normal | Verbosity::Detailed)
    }

    /// Returns true if detailed verbosity is set
    pub fn detailed_verbosity(&self) -> bool {
        matches!(self, Verbosity::Detailed)
    }
}

/// Parse verbosity leverl
///
/// ### Params
///
/// * `level` - If `1` returns [Verbosity::Normal], with `2`
///   [Verbosity::Detailed]
///
/// ### Returns
///
/// The desired [Verbosity] level.
pub fn parse_verbosity_level(level: usize) -> Verbosity {
    match level {
        0 => Verbosity::Quiet,
        1 => Verbosity::Normal,
        2 => Verbosity::Detailed,
        _ => Verbosity::Quiet,
    }
}

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
