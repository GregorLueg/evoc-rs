//! Persistence-based multi-layer cluster extraction for HDBSCAN.
//!
//! Computes a barcode of cluster lifetimes across `min_cluster_size` values,
//! finds peaks in the resulting total-persistence curve, and extracts one
//! cluster labelling per diverse peak. This produces multiple clustering
//! "layers" at different resolution levels, sorted by decreasing cluster count.

use crate::clustering::condensed_tree::*;
use crate::prelude::*;

/// Per-cluster birth/death information for persistence analysis.
///
/// Describes the range of `min_cluster_size` thresholds for which a given
/// cluster node exists in the condensed tree.
#[derive(Clone, Debug)]
pub struct ClusterBarcode<T> {
    /// min_cluster_size at which this cluster is "born" (appears)
    pub size_birth: T,
    /// min_cluster_size at which this cluster "dies" (merges away)
    pub size_death: T,
    /// parent cluster index
    pub parent: usize,
    /// lambda value at death
    pub lambda_death: T,
}

/// Compute the `min_cluster_size` barcode for all cluster nodes.
///
/// Each entry describes the birth and death sizes for one cluster node.
/// Index `i` corresponds to cluster node `n_samples + i`. Cluster nodes
/// are processed bottom-up in pairs (each genuine split produces exactly
/// two sibling children). The root occupies index 0 with `size_death = n_samples`.
///
/// ### Params
///
/// * `tree` - Condensed tree produced by `condense_tree()`
///
/// ### Returns
///
/// One `ClusterBarcode` per cluster node, in order of node index; empty if the
/// tree has no cluster-to-cluster edges
pub fn min_cluster_size_barcode<T>(tree: &CondensedTree<T>) -> Vec<ClusterBarcode<T>>
where
    T: EvocFloat,
{
    let n_points = tree.n_samples;
    if tree.nodes.is_empty() {
        return Vec::new();
    }

    // find all cluster-to-cluster edges (child_size > 1)
    let cluster_nodes: Vec<&CondensedNode<T>> =
        tree.nodes.iter().filter(|n| n.child_size > 1).collect();

    if cluster_nodes.is_empty() {
        return Vec::new();
    }

    let max_child = cluster_nodes.iter().map(|n| n.child).max().unwrap();
    let n_nodes = max_child - n_points + 1;

    let mut barcodes = vec![
        ClusterBarcode {
            size_birth: T::from(2).unwrap(),
            size_death: T::zero(),
            parent: n_points,
            lambda_death: T::zero(),
        };
        n_nodes
    ];

    // root node
    barcodes[0].lambda_death = T::zero();
    barcodes[0].size_death = T::from(n_points).unwrap();

    // process cluster pairs (they come in pairs: two children per split)
    // iterate in reverse to process bottom-up
    let mut i = cluster_nodes.len();
    while i >= 2 {
        i -= 2;
        let node_a = cluster_nodes[i];
        let node_b = cluster_nodes[i + 1];

        let idx_a = node_a.child - n_points;
        let idx_b = node_b.child - n_points;

        barcodes[idx_a].parent = node_a.parent;
        barcodes[idx_b].parent = node_b.parent;

        // lambda at death = exp(-1/lambda_val) in the Python, but we
        // store the raw lambda for now (the persistence computation
        // uses it directly)
        let lv = node_a.lambda_val;
        barcodes[idx_a].lambda_death = lv;
        barcodes[idx_b].lambda_death = lv;

        let death_size = node_a.child_size.min(node_b.child_size);
        let death_t = T::from(death_size).unwrap();
        barcodes[idx_a].size_death = death_t;
        barcodes[idx_b].size_death = death_t;

        let parent_idx = node_a.parent - n_points;
        let max_birth = barcodes[idx_a]
            .size_birth
            .max(barcodes[idx_b].size_birth)
            .max(death_t);
        barcodes[parent_idx].size_birth = barcodes[parent_idx].size_birth.max(max_birth);
    }

    barcodes
}

/// Compute total persistence as a function of `min_cluster_size`.
///
/// Each cluster contributes `(size_death - size_birth) * lambda_death`
/// to every size bin it spans. The result is a discretised persistence
/// curve over the unique birth-size values seen in the barcodes.
///
/// ### Params
///
/// * `barcodes` - Output of `min_cluster_size_barcode()`
///
/// ### Returns
///
/// `(sizes, persistence)` where `sizes[i]` is a unique `min_cluster_size`
/// threshold and `persistence[i]` is the total persistence at that threshold;
/// both vecs are empty if `barcodes` is empty
pub fn compute_total_persistence<T>(barcodes: &[ClusterBarcode<T>]) -> (Vec<T>, Vec<T>)
where
    T: EvocFloat,
{
    if barcodes.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut size_set: Vec<T> = barcodes.iter().map(|b| b.size_birth).collect();
    size_set.sort_by(|a, b| a.partial_cmp(b).unwrap());
    size_set.dedup();

    let mut persistence = vec![T::zero(); size_set.len()];

    for (bc_idx, bc) in barcodes.iter().enumerate() {
        if bc_idx == 0 {
            continue; // skip root
        }
        if bc.size_death <= bc.size_birth {
            continue;
        }

        let birth_pos = size_set
            .iter()
            .position(|&s| s >= bc.size_birth)
            .unwrap_or(size_set.len());
        let death_pos = size_set
            .iter()
            .position(|&s| s >= bc.size_death)
            .unwrap_or(size_set.len());

        let contribution = (bc.size_death - bc.size_birth) * bc.lambda_death;
        for k in birth_pos..death_pos {
            persistence[k] += contribution;
        }
    }

    (size_set, persistence)
}

/// Find local maxima in a 1-D signal.
///
/// Plateau peaks are reported at their midpoint index. Matches the behaviour
/// of `scipy.signal.find_peaks` with no additional constraints.
///
/// ### Params
///
/// * `x` - Signal values
///
/// ### Returns
///
/// Indices of local maxima in ascending order; empty if `x` has fewer than
/// three elements or contains no peaks
pub fn find_peaks<T>(x: &[T]) -> Vec<usize>
where
    T: EvocFloat,
{
    if x.len() < 3 {
        return Vec::new();
    }

    let mut peaks = Vec::new();
    let mut i = 1;
    let i_max = x.len() - 1;

    while i < i_max {
        if x[i - 1] < x[i] {
            let mut i_ahead = i + 1;
            while i_ahead < i_max && x[i_ahead] == x[i] {
                i_ahead += 1;
            }
            if x[i_ahead] < x[i] {
                peaks.push((i + i_ahead - 1) / 2);
                i = i_ahead;
            }
        }
        i += 1;
    }
    peaks
}

/// Estimate the Jaccard similarity between the clusterings active at two
/// different birth-size thresholds.
///
/// A cluster is "active" at a given size if `size_birth <= size <= size_death`.
/// Similarity is computed over barcode indices, not point assignments.
///
/// ### Params
///
/// * `barcodes` - Output of `min_cluster_size_barcode()`
/// * `birth_a` - First birth-size threshold
/// * `birth_b` - Second birth-size threshold
///
/// ### Returns
///
/// Jaccard similarity in `[0, 1]`; `0` if both active sets are empty
fn estimate_cluster_similarity<T>(barcodes: &[ClusterBarcode<T>], birth_a: T, birth_b: T) -> f64
where
    T: EvocFloat,
{
    let active_a: std::collections::HashSet<usize> = barcodes
        .iter()
        .enumerate()
        .filter(|(_, b)| b.size_birth <= birth_a && b.size_death > birth_a)
        .map(|(i, _)| i)
        .collect();

    let active_b: std::collections::HashSet<usize> = barcodes
        .iter()
        .enumerate()
        .filter(|(_, b)| b.size_birth <= birth_b && b.size_death > birth_b)
        .map(|(i, _)| i)
        .collect();

    let intersection = active_a.intersection(&active_b).count();
    let union = active_a.union(&active_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Select a diverse subset of persistence-curve peaks.
///
/// Greedily picks peaks in descending persistence order, skipping any whose
/// clustering (at that birth size) is too similar (by Jaccard index) to a peak
/// already selected.
///
/// ### Params
///
/// * `peaks` - Peak indices into `persistence` and `sizes`, from `find_peaks()`
/// * `persistence` - Persistence curve values
/// * `sizes` - Corresponding `min_cluster_size` values
/// * `barcodes` - Output of `min_cluster_size_barcode()`
/// * `min_similarity_threshold` - Maximum Jaccard similarity allowed between
///   any two selected peaks; peaks above this are skipped
/// * `max_layers` - Upper bound on the number of peaks to select
///
/// ### Returns
///
/// Selected peak indices (into `persistence`/`sizes`) in descending
/// persistence order
pub fn select_diverse_peaks<T>(
    peaks: &[usize],
    persistence: &[T],
    sizes: &[T],
    barcodes: &[ClusterBarcode<T>],
    min_similarity_threshold: f64,
    max_layers: usize,
) -> Vec<usize>
where
    T: EvocFloat,
{
    if peaks.is_empty() {
        return Vec::new();
    }

    // sort peaks by descending persistence
    let mut indexed: Vec<(usize, T)> = peaks.iter().map(|&p| (p, persistence[p])).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut selected = Vec::new();
    let mut selected_births = Vec::new();

    for (peak, _) in indexed {
        if selected.len() >= max_layers {
            break;
        }

        let birth = sizes[peak];
        let is_diverse = selected_births.iter().all(|&sb| {
            estimate_cluster_similarity(barcodes, birth, sb) <= min_similarity_threshold
        });

        if is_diverse {
            selected.push(peak);
            selected_births.push(birth);
        }
    }

    selected
}

/// Extract point labels and membership strengths at a specific birth-size
/// threshold.
///
/// Clusters active at `birth_size`
/// (i.e. `size_birth <= birth_size < size_death`) are selected from the
/// barcodes. Relies on the invariant that `barcodes[i]` corresponds to cluster
/// node `n_samples + i`.
///
/// ### Params
///
/// * `tree` - Condensed tree produced by `condense_tree()`
/// * `barcodes` - Output of `min_cluster_size_barcode()`
/// * `birth_size` - The `min_cluster_size` threshold to evaluate at
/// * `n_samples` - Number of original data points
///
/// ### Returns
///
/// `(labels, strengths)` — cluster label per point (`-1` = noise) and
/// membership strength in `[0, 1]` per point
pub fn extract_clusters_at_size<T>(
    tree: &CondensedTree<T>,
    barcodes: &[ClusterBarcode<T>],
    birth_size: T,
    n_samples: usize,
) -> (Vec<i64>, Vec<T>)
where
    T: EvocFloat,
{
    let active_clusters: Vec<usize> = barcodes
        .iter()
        .enumerate()
        .filter(|(_, b)| b.size_birth <= birth_size && b.size_death > birth_size)
        .map(|(i, _)| i + n_samples)
        .collect();

    let labels = get_cluster_label_vector(tree, &active_clusters, n_samples);
    let strengths = get_point_membership_strengths(tree, &active_clusters, &labels);

    (labels, strengths)
}

/// Build all cluster layers from an embedding via persistence-guided HDBSCAN.
///
/// Runs the full pipeline: MST -> linkage -> condensed tree -> barcode ->
/// persistence peaks -> per-peak labelling. The base layer (using
/// `base_min_cluster_size` directly) is always included. Additional layers
/// are drawn from diverse peaks in the persistence curve, up to `max_layers`
/// total. Layers are returned sorted by descending cluster count (finest
/// granularity first).
///
/// ### Params
///
/// * `embedding` - Point coordinates; typically the EVoC embedding
/// * `min_samples` - Neighbourhood size for core-distance computation in the
///   MST
/// * `base_min_cluster_size` - Minimum cluster size for the base condensed tree
/// * `min_similarity_threshold` - Maximum Jaccard similarity between selected
///   persistence peaks; controls diversity of layers
/// * `max_layers` - Maximum total number of layers to return, including the
///   base
///
/// ### Returns
///
/// `(labels, strengths, persistence)` — parallel vecs of length ≤ `max_layers`;
/// each entry is one clustering layer. `persistence[0]` is always `0.0` for the
/// base layer; remaining entries are the raw persistence values of the selected
/// peaks
pub fn build_cluster_layers<T>(
    embedding: &[Vec<T>],
    min_samples: usize,
    base_min_cluster_size: usize,
    min_similarity_threshold: f64,
    max_layers: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<T>>, Vec<f64>)
where
    T: EvocFloat,
{
    let n = embedding.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut mst = crate::clustering::mst::build_mst(embedding, min_samples);
    let linkage = crate::clustering::linkage::mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, base_min_cluster_size);

    let leaves = extract_leaves(&ct);
    let base_labels = get_cluster_label_vector(&ct, &leaves, n);
    let base_strengths = get_point_membership_strengths(&ct, &leaves, &base_labels);

    let mut all_labels = vec![base_labels];
    let mut all_strengths = vec![base_strengths];
    let mut all_persistence = vec![0.0f64];

    let barcodes = min_cluster_size_barcode(&ct);
    if !barcodes.is_empty() {
        let (sizes, persistence) = compute_total_persistence(&barcodes);
        let peaks = find_peaks(&persistence);

        let selected = select_diverse_peaks(
            &peaks,
            &persistence,
            &sizes,
            &barcodes,
            min_similarity_threshold,
            max_layers.saturating_sub(1),
        );

        for &peak in &selected {
            let birth_size = sizes[peak];
            let (labels, strengths) = extract_clusters_at_size(&ct, &barcodes, birth_size, n);
            let p = persistence[peak].to_f64().unwrap_or(0.0);
            all_labels.push(labels);
            all_strengths.push(strengths);
            all_persistence.push(p);
        }
    }

    // sort by number of clusters, descending (finest first)
    let mut order: Vec<usize> = (0..all_labels.len()).collect();
    order.sort_by(|&a, &b| {
        let na = all_labels[a]
            .iter()
            .filter(|&&l| l >= 0)
            .max()
            .unwrap_or(&-1)
            + 1;
        let nb = all_labels[b]
            .iter()
            .filter(|&&l| l >= 0)
            .max()
            .unwrap_or(&-1)
            + 1;
        nb.cmp(&na)
    });

    let labels_sorted = order.iter().map(|&i| all_labels[i].clone()).collect();
    let strengths_sorted = order.iter().map(|&i| all_strengths[i].clone()).collect();
    let persistence_sorted = order.iter().map(|&i| all_persistence[i]).collect();

    (labels_sorted, strengths_sorted, persistence_sorted)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::condensed_tree::condense_tree;
    use crate::clustering::linkage::mst_to_linkage_tree;
    use crate::clustering::mst::build_mst;
    use num_traits::ToPrimitive;

    fn two_cluster_data() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![10.1, 10.1],
            vec![10.05, 10.05],
        ]
    }

    fn three_cluster_data() -> Vec<Vec<f64>> {
        let mut data = two_cluster_data();
        data.extend(vec![
            vec![20.0, 0.0],
            vec![20.1, 0.0],
            vec![20.0, 0.1],
            vec![20.1, 0.1],
            vec![20.05, 0.05],
        ]);
        data
    }

    #[test]
    fn test_find_peaks_basic() {
        let x = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let peaks = find_peaks(&x);
        assert_eq!(peaks, vec![1, 3]);
    }

    #[test]
    fn test_find_peaks_plateau() {
        let x = vec![0.0, 1.0, 1.0, 0.0];
        let peaks = find_peaks(&x);
        assert_eq!(peaks.len(), 1);
        assert!(peaks[0] == 1 || peaks[0] == 2); // midpoint
    }

    #[test]
    fn test_find_peaks_monotonic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let peaks = find_peaks(&x);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_find_peaks_empty() {
        let peaks = find_peaks::<f64>(&[]);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_find_peaks_short() {
        let peaks = find_peaks(&[1.0, 2.0]);
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_barcode_non_empty() {
        let data = two_cluster_data();
        let n = data.len();
        let mut mst = build_mst(&data, 1);
        let linkage = mst_to_linkage_tree(&mut mst, n);
        let ct = condense_tree(&linkage, n, 2);
        let barcodes = min_cluster_size_barcode(&ct);

        assert!(!barcodes.is_empty());
        // root should have death = n
        assert!(barcodes[0].size_death.to_usize().unwrap() == n);
    }

    #[test]
    fn test_persistence_shape() {
        let data = two_cluster_data();
        let n = data.len();
        let mut mst = build_mst(&data, 1);
        let linkage = mst_to_linkage_tree(&mut mst, n);
        let ct = condense_tree(&linkage, n, 2);
        let barcodes = min_cluster_size_barcode(&ct);
        let (sizes, persistence) = compute_total_persistence(&barcodes);

        assert_eq!(sizes.len(), persistence.len());
        assert!(!sizes.is_empty());
        // persistence values should be non-negative
        assert!(persistence.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_build_cluster_layers_two_clusters() {
        let data = two_cluster_data();
        let (labels, strengths, persistence) = build_cluster_layers(&data, 1, 2, 0.2, 10);

        assert!(!labels.is_empty());
        assert_eq!(labels.len(), strengths.len());
        assert_eq!(labels.len(), persistence.len());

        // base layer should find 2 clusters
        let base = &labels[0];
        let n_clusters = base
            .iter()
            .filter(|&&l| l >= 0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        assert!(
            n_clusters >= 2,
            "expected at least 2 clusters, got {}",
            n_clusters
        );

        // first 5 points should be in one cluster, last 5 in another
        assert_eq!(base[0], base[1]);
        assert_eq!(base[0], base[4]);
        assert_eq!(base[5], base[6]);
        assert_eq!(base[5], base[9]);
        assert_ne!(base[0], base[5]);
    }

    #[test]
    fn test_build_cluster_layers_three_clusters() {
        let data = three_cluster_data();
        let (labels, _, _) = build_cluster_layers(&data, 1, 2, 0.2, 10);

        let base = &labels[0];
        let n_clusters = base
            .iter()
            .filter(|&&l| l >= 0)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        assert!(
            n_clusters >= 3,
            "expected at least 3 clusters, got {}",
            n_clusters
        );
    }

    #[test]
    fn test_build_cluster_layers_empty() {
        let data: Vec<Vec<f64>> = Vec::new();
        let (labels, strengths, persistence) = build_cluster_layers(&data, 1, 2, 0.2, 10);
        assert!(labels.is_empty());
        assert!(strengths.is_empty());
        assert!(persistence.is_empty());
    }

    #[test]
    fn test_layers_sorted_by_cluster_count() {
        let data = three_cluster_data();
        let (labels, _, _) = build_cluster_layers(&data, 1, 2, 0.2, 10);

        if labels.len() >= 2 {
            let count = |l: &Vec<i64>| -> i64 {
                l.iter().filter(|&&v| v >= 0).max().copied().unwrap_or(-1) + 1
            };
            for i in 1..labels.len() {
                assert!(
                    count(&labels[i - 1]) >= count(&labels[i]),
                    "layers not sorted descending by cluster count"
                );
            }
        }
    }

    #[test]
    fn test_strengths_in_valid_range() {
        let data = two_cluster_data();
        let (labels, strengths, _) = build_cluster_layers(&data, 1, 2, 0.2, 10);

        for (layer_labels, layer_strengths) in labels.iter().zip(strengths.iter()) {
            for (i, (&label, &strength)) in
                layer_labels.iter().zip(layer_strengths.iter()).enumerate()
            {
                if label >= 0 {
                    assert!(
                        (0.0..=1.0).contains(&strength),
                        "point {} in layer has strength {} outside [0,1]",
                        i,
                        strength
                    );
                }
            }
        }
    }

    #[test]
    fn test_diverse_peaks_respects_max_layers() {
        let data = three_cluster_data();
        let (labels, _, _) = build_cluster_layers(&data, 1, 2, 0.2, 2);
        assert!(labels.len() <= 2, "should respect max_layers=2");
    }

    #[test]
    fn test_select_diverse_peaks_empty() {
        let result = select_diverse_peaks::<f64>(&[], &[], &[], &[], 0.2, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_select_diverse_peaks_single() {
        let persistence = vec![0.0, 5.0, 0.0];
        let sizes = vec![2.0, 3.0, 4.0];
        let barcodes = vec![
            ClusterBarcode {
                size_birth: 2.0,
                size_death: 10.0,
                parent: 0,
                lambda_death: 1.0,
            },
            ClusterBarcode {
                size_birth: 2.0,
                size_death: 5.0,
                parent: 0,
                lambda_death: 0.5,
            },
        ];
        let peaks = vec![1];
        let result = select_diverse_peaks(&peaks, &persistence, &sizes, &barcodes, 0.2, 10);
        assert_eq!(result, vec![1]);
    }
}
