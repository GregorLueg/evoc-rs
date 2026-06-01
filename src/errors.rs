//! Errors for the evoc-rs crate.

use thiserror::Error;

/// Errors that can be returned by manifolds-rs
#[derive(Debug, Error)]
pub enum EvocErrors {
    // -- manifolds-rs --
    /// Propagate errors from the ann-search-rs crate
    #[error("Error from the ann-search-rs crate: {0}")]
    AnnSearchRsError(#[from] ann_search_rs::errors::AnnSearchErrors),
}
