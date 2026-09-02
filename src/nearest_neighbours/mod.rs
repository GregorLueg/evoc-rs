//! Nearest neighbour algorithms from `ann-search-rs`

pub mod nearest_neighbour_cpu;
#[cfg(feature = "gpu")]
pub mod nearest_neighbour_gpu;

use num_traits::Float;
use rayon::prelude::*;

/////////////////////
// Distance rescue //
/////////////////////

/// Whether a metric name reaches `ann-search-rs` as a squared distance.
///
/// `parse_ann_dist` maps both `"euclidean"` and `"l2"` to
/// `Dist::SquaredEuclidean`, so the backends never take the square root. Every
/// other metric is returned on its own scale.
///
/// ### Params
///
/// * `dist_metric` - Metric name as handed to the backend.
///
/// ### Returns
///
/// `true` when the returned distances are squared Euclidean.
fn is_squared_euclidean(dist_metric: &str) -> bool {
    matches!(dist_metric.to_lowercase().as_str(), "euclidean" | "l2")
}

/// Put Euclidean distances back on a true Euclidean scale.
///
/// The backends return squared Euclidean for `"euclidean"` and `"l2"`, which
/// leaves the crate handing out one metric on a squared scale and every other
/// on a linear one. Neighbour ordering is unaffected either way, since the
/// square root is monotonic on non-negative values, but the magnitudes are not:
/// the fuzzy simplicial set fits its bandwidths to these numbers, and callers
/// reading the returned graph expect the distance the metric is named after.
///
/// A no-op for every metric that is not Euclidean.
///
/// ### Params
///
/// * `knn_dist` - Per-point neighbour distances, as the backend returned them.
/// * `dist_metric` - Metric name the search ran with.
///
/// ### Returns
///
/// The distances, square-rooted when the metric was Euclidean.
pub(crate) fn rescale_distances<T>(knn_dist: Vec<Vec<T>>, dist_metric: &str) -> Vec<Vec<T>>
where
    T: Float + Send,
{
    if !is_squared_euclidean(dist_metric) {
        return knn_dist;
    }

    knn_dist
        .into_par_iter()
        .map(|row| {
            row.into_iter()
                // Clamped rather than trusted: a backend can return a small
                // negative from rounding, and `sqrt` of that is NaN, which
                // would poison the bandwidth search downstream.
                .map(|d| if d > T::zero() { d.sqrt() } else { T::zero() })
                .collect()
        })
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_and_l2_are_squared() {
        assert!(is_squared_euclidean("euclidean"));
        assert!(is_squared_euclidean("l2"));
        assert!(is_squared_euclidean("Euclidean"));
        assert!(!is_squared_euclidean("cosine"));
        assert!(!is_squared_euclidean("manhattan"));
    }

    #[test]
    fn test_euclidean_distances_are_rooted() {
        let dist = vec![vec![4.0f64, 9.0], vec![16.0, 25.0]];
        let out = rescale_distances(dist, "euclidean");

        assert_relative_eq!(out[0][0], 2.0);
        assert_relative_eq!(out[0][1], 3.0);
        assert_relative_eq!(out[1][0], 4.0);
        assert_relative_eq!(out[1][1], 5.0);
    }

    #[test]
    fn test_cosine_distances_pass_through() {
        let dist = vec![vec![0.25f64, 0.5]];
        let out = rescale_distances(dist.clone(), "cosine");

        assert_eq!(out, dist);
    }

    #[test]
    fn test_negative_rounding_clamps_to_zero() {
        let dist = vec![vec![-1e-16f64, 0.0, 4.0]];
        let out = rescale_distances(dist, "l2");

        assert_relative_eq!(out[0][0], 0.0);
        assert_relative_eq!(out[0][1], 0.0);
        assert_relative_eq!(out[0][2], 2.0);
    }

    #[test]
    fn test_ordering_is_preserved() {
        let dist = vec![vec![1.0f64, 4.0, 9.0, 16.0]];
        let out = rescale_distances(dist, "euclidean");

        assert!(out[0].windows(2).all(|w| w[0] <= w[1]));
    }
}
