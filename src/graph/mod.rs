//! This module contains the needed graph-related functions: kNN graph
//! generation, fuzzy graph generation, label propagation and the embedding
//! optimisation.

pub mod embedding;
pub mod fuzzy_graph;
pub mod label_prop;

use crate::prelude::*;

////////////
// Params //
////////////

/// Default beta1 value for Adam optimisation (Umap-specific)
pub const UMAP_BETA1: f64 = 0.5;
/// Default beta2 value for Adam optimisation (Umap-specific)
pub const UMAP_BETA2: f64 = 0.9;

/// Parameters for the EVoC node embedding
#[derive(Clone, Debug)]
pub struct EvocEmbeddingParams<T> {
    /// Number of iterations during the label propagation
    pub n_label_prop_iter: usize,
    /// Base initialisation threshold
    pub base_init_threshold: usize,
    /// Scaling parameter
    pub scaling: T,
    /// Number of training epochs. Higher values give more refined embeddings at
    /// the cost of runtime. Default: 50.
    pub n_epochs: usize,
    /// Scaling factor in the attractive gradient kernel numerator (
    /// `-2 * noise_level * d - 2`). Controls how aggressively connected nodes
    /// attract. Default: 0.5.
    pub noise_level: T,
    /// Controls how many negative (repulsive) samples are drawn per positive
    /// edge sample. Lower values reduce repulsion; setting to zero effectively
    /// disables it. Default: 1.0.
    pub negative_sample_rate: T,
    /// Initial learning rate, linearly decayed to zero over training. Default:
    /// 0.1.
    pub initial_alpha: T,
    /// Adam first moment decay rate. Default: 0.5.
    pub beta1: T,
    /// Adam second moment decay rate. Default: 0.9.
    pub beta2: T,
    /// Adam numerical stability constant. Default: 1e-7.
    pub eps: T,
}

/// Defaults for the EvocEmbeddingParameters
impl<T: EvocFloat> Default for EvocEmbeddingParams<T> {
    fn default() -> Self {
        Self {
            n_label_prop_iter: 20,
            base_init_threshold: 64,
            scaling: T::from(0.1).unwrap(),
            n_epochs: 50,
            noise_level: T::from(0.5).unwrap(),
            negative_sample_rate: T::one(),
            initial_alpha: T::from(0.1).unwrap(),
            beta1: T::from(UMAP_BETA1).unwrap(),
            beta2: T::from(UMAP_BETA2).unwrap(),
            eps: T::from(1e-7).unwrap(),
        }
    }
}
