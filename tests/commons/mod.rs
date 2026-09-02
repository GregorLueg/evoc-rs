use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Generate `n_clusters` isotropic Gaussian blobs in `dim` dimensions.
///
/// Cluster `c` is centred at `(c * separation, 0, ..., 0)` and every coordinate
/// carries independent Gaussian noise of standard deviation `spread`.
///
/// The centre offset is one-dimensional while the noise is `dim`-dimensional,
/// so the typical within-cluster distance is `spread * sqrt(2 * dim)` while the
/// adjacent-centre distance stays at `separation`. Separability therefore
/// degrades as `dim` grows for a fixed `separation`. Keep `separation` well
/// above `spread * sqrt(2 * dim)`.
///
/// Use this for the tests that consume the raw coordinates (MST, linkage,
/// condensed tree, fuzzy graph). It is a poor fixture for end-to-end `evoc`
/// accuracy: an isotropic blob embeds to a near-uniform density, the
/// persistence extraction has nothing to lock onto, and the selected layer
/// fragments unpredictably with the data seed. Use
/// `ann_search_rs::synthetic::generate_clustered_data` there.
///
/// ### Params
///
/// * `n_per_cluster` - Points drawn per cluster.
/// * `n_clusters` - Number of clusters.
/// * `dim` - Dimensionality of each point.
/// * `separation` - Distance between adjacent cluster centres along axis 0.
/// * `spread` - Per-coordinate standard deviation of the Gaussian noise.
/// * `seed` - Random seed for reproducibility.
///
/// ### Returns
///
/// `(data, labels)` where `data[i]` is a point and `labels[i]` its
/// ground-truth cluster index.
pub fn make_blobs(
    n_per_cluster: usize,
    n_clusters: usize,
    dim: usize,
    separation: f64,
    spread: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(n_per_cluster * n_clusters);
    let mut labels = Vec::with_capacity(n_per_cluster * n_clusters);

    for c in 0..n_clusters {
        for _ in 0..n_per_cluster {
            let mut point = vec![0.0f64; dim];
            // Centre along first axis, spread in all dimensions
            point[0] = c as f64 * separation;
            #[allow(clippy::needless_range_loop)]
            for d in 0..dim {
                // Box-Muller for normal samples
                let u1: f64 = rng.random_range(1e-10..1.0);
                let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                point[d] += spread * (-2.0 * u1.ln()).sqrt() * u2.cos();
            }
            data.push(point);
            labels.push(c);
        }
    }
    (data, labels)
}

/// Count the number of distinct non-noise clusters in a label vector.
///
/// ### Params
///
/// * `labels` - Cluster labels, `-1` for noise.
///
/// ### Returns
///
/// The number of non-noise clusters.
#[allow(dead_code)] // clippy being stupid
pub fn count_clusters(labels: &[i64]) -> usize {
    labels
        .iter()
        .filter(|&&l| l >= 0)
        .max()
        .map(|&m| m as usize + 1)
        .unwrap_or(0)
}

/// Compute the fraction of points whose assigned cluster label agrees with
/// ground truth (up to a permutation of label indices).
///
/// Uses a simple majority-vote alignment: for each predicted cluster, find
/// the most common ground-truth label and count agreements.
///
/// This is purity, not a matched accuracy. Several predicted clusters may map
/// to the same ground-truth label, so over-segmentation is not penalised;
/// noise points are counted as wrong, so a conservative clustering is.
///
/// ### Params
///
/// * `predicted` - Predicted cluster labels, `-1` for noise.
/// * `ground_truth` - True cluster index per point.
///
/// ### Returns
///
/// The fraction of points in `[0, 1]` assigned to a cluster whose majority
/// ground-truth label matches their own.
#[allow(dead_code)] // used by integration_tests only
pub fn cluster_accuracy(predicted: &[i64], ground_truth: &[usize]) -> f64 {
    use std::collections::HashMap;

    let n = predicted.len();
    if n == 0 {
        return 1.0;
    }

    // For each predicted cluster, find the dominant ground-truth label
    let mut cluster_votes: HashMap<i64, HashMap<usize, usize>> = HashMap::new();
    for (i, &pred) in predicted.iter().enumerate() {
        if pred < 0 {
            continue;
        }
        *cluster_votes
            .entry(pred)
            .or_default()
            .entry(ground_truth[i])
            .or_default() += 1;
    }

    // Greedy assignment: each predicted cluster maps to its majority gt label
    let mut correct = 0usize;
    for votes in cluster_votes.values() {
        correct += votes.values().max().copied().unwrap_or(0);
    }

    correct as f64 / n as f64
}
