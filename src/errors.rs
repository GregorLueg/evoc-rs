//! Errors for the evoc-rs crate.

use manifolds_rs::prelude::ManifoldsError;
use thiserror::Error;

/// Errors that can be returned by manifolds-rs
#[derive(Debug, Error)]
pub enum EvocErrors {
    // -- manifolds-rs --
    /// Propagate errors from the manifolds-rs crate
    #[error("Error from the manifolds-rs crate: {0}")]
    AnnSearchRsError(#[from] ManifoldsError),
}
