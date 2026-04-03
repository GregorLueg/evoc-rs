//! Condensed tree construction and cluster extraction for HDBSCAN.
//!
//! Converts a single-linkage dendrogram into a condensed tree by collapsing
//! splits where either child is smaller than `min_cluster_size`. The result
//! tracks how clusters persist across the lambda (1/distance) scale. Leaf
//! clusters in the condensed tree are the final cluster candidates; points
//! not belonging to any selected cluster are labelled noise (-1).

use crate::clustering::linkage::LinkageRow;
use crate::prelude::*;
use crate::utils::disjoint_set::DisjointSet;

////////////////
// Structures //
////////////////

/// A single node in the condensed cluster tree.
#[derive(Clone, Debug)]
pub struct CondensedNode<T> {
    /// Cluster this node belongs to (internal node index >= n_samples)
    pub parent: usize,
    /// Either a point index (< n_samples) or a child cluster index
    pub child: usize,
    /// Lambda (1 / distance) at which this child falls out of the parent
    pub lambda_val: T,
    /// Number of original points under this child (1 for leaf points)
    pub child_size: usize,
}

/// The condensed cluster hierarchy produced by [`condense_tree`].
#[derive(Clone, Debug)]
pub struct CondensedTree<T> {
    /// All nodes in the condensed tree, in traversal order
    pub nodes: Vec<CondensedNode<T>>,
    /// Number of original data points (leaf node indices are 0..n_samples)
    pub n_samples: usize,
}

/// Collect all descendants of `root` in the linkage hierarchy (DFS order).
///
/// Returns both internal nodes (index >= n_samples) and leaf points
/// (index < n_samples).
///
/// ### Params
///
/// * `linkage` - Linkage matrix. Slice of `LinkageRow<T>`
/// * `root` - Node index to start traversal from
/// * `n_samples` - Number of original data points; used to distinguish leaves
///   from internal nodes
///
/// ### Returns
///
/// All descendant node indices in DFS order, including `root` itself
fn collect_descendants<T>(linkage: &[LinkageRow<T>], root: usize, n_samples: usize) -> Vec<usize>
where
    T: EvocFloat,
{
    let mut result = Vec::new();
    let mut queue = vec![root];

    while let Some(node) = queue.pop() {
        result.push(node);
        if node >= n_samples {
            let row = &linkage[node - n_samples];
            queue.push(row.left);
            queue.push(row.right);
        }
    }
    result
}

/// Flatten all leaf points under `node` into the condensed tree as direct
/// children of `parent_cluster` at `lambda_val`.
///
/// Used when a subtree is too small to form its own cluster; its points are
/// attributed to the surviving parent cluster instead.
///
/// ### Params
///
/// * `linkage` - Linkage matrix. Slice of `LinkageRow<T>`
/// * `node` - Root of the subtree being eliminated
/// * `parent_cluster` - Cluster index the eliminated points are assigned to
/// * `lambda_val` - Lambda at which the elimination occurs
/// * `n_samples` - Number of original data points
/// * `output` - Condensed tree node buffer, appended to in place
fn eliminate_branch<T>(
    linkage: &[LinkageRow<T>],
    node: usize,
    parent_cluster: usize,
    lambda_val: T,
    n_samples: usize,
    output: &mut Vec<CondensedNode<T>>,
) where
    T: EvocFloat,
{
    let descendants = collect_descendants(linkage, node, n_samples);
    for d in descendants {
        if d < n_samples {
            output.push(CondensedNode {
                parent: parent_cluster,
                child: d,
                lambda_val,
                child_size: 1,
            });
        }
    }
}

/// Condense a linkage tree with a given `min_cluster_size`.
///
/// Walks the hierarchy top-down. When a split produces a child smaller
/// than `min_cluster_size`, that branch is eliminated and its points are
/// assigned to the parent cluster. Only splits where both children meet
/// the minimum size produce new cluster nodes.
///
/// ### Params
///
/// * `linkage` - Linkage matrix. Slice of `LinkageRow<T>`
/// * `n_samples` - Number of original data points
/// * `min_cluster_size` - Minimum number of points for a split to produce a new
///   cluster
///
/// ### Returns
///
/// A [`CondensedTree`] whose leaf clusters are candidates for final cluster selection
pub fn condense_tree<T>(
    linkage: &[LinkageRow<T>],
    n_samples: usize,
    min_cluster_size: usize,
) -> CondensedTree<T>
where
    T: EvocFloat,
{
    if linkage.is_empty() {
        return CondensedTree {
            nodes: Vec::new(),
            n_samples,
        };
    }

    let root = 2 * linkage.len(); // == 2*n_samples - 2, the root node index

    // BFS traversal order from root
    let traversal = collect_descendants(linkage, root, n_samples);

    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n_samples; // root cluster gets label n_samples

    let mut next_label = n_samples + 1;
    let mut ignore = vec![false; root + 1];
    let mut output = Vec::new();

    for &node in &traversal {
        if node < n_samples || ignore[node] {
            continue;
        }

        let parent_label = relabel[node];
        let row = &linkage[node - n_samples];
        let left = row.left;
        let right = row.right;
        let distance = row.distance;

        let lambda = if distance > T::zero() {
            T::one() / distance
        } else {
            T::infinity()
        };

        let left_size = if left >= n_samples {
            linkage[left - n_samples].size
        } else {
            1
        };
        let right_size = if right >= n_samples {
            linkage[right - n_samples].size
        } else {
            1
        };

        let left_big = left_size >= min_cluster_size;
        let right_big = right_size >= min_cluster_size;

        match (left_big, right_big) {
            (true, true) => {
                // genuine split: both children become new clusters
                relabel[left] = next_label;
                output.push(CondensedNode {
                    parent: parent_label,
                    child: next_label,
                    lambda_val: lambda,
                    child_size: left_size,
                });
                next_label += 1;

                relabel[right] = next_label;
                output.push(CondensedNode {
                    parent: parent_label,
                    child: next_label,
                    lambda_val: lambda,
                    child_size: right_size,
                });
                next_label += 1;
            }
            (true, false) => {
                // right is too small: eliminate right, continue left
                relabel[left] = parent_label;
                eliminate_branch(linkage, right, parent_label, lambda, n_samples, &mut output);
                // mark internal nodes in right branch as ignored
                for d in collect_descendants(linkage, right, n_samples) {
                    if d >= n_samples {
                        ignore[d] = true;
                    }
                }
            }
            (false, true) => {
                // left is too small: eliminate left, continue right
                relabel[right] = parent_label;
                eliminate_branch(linkage, left, parent_label, lambda, n_samples, &mut output);
                for d in collect_descendants(linkage, left, n_samples) {
                    if d >= n_samples {
                        ignore[d] = true;
                    }
                }
            }
            (false, false) => {
                // both too small: eliminate both
                eliminate_branch(linkage, left, parent_label, lambda, n_samples, &mut output);
                eliminate_branch(linkage, right, parent_label, lambda, n_samples, &mut output);
                for d in collect_descendants(linkage, left, n_samples)
                    .into_iter()
                    .chain(collect_descendants(linkage, right, n_samples))
                {
                    if d >= n_samples {
                        ignore[d] = true;
                    }
                }
            }
        }
    }

    CondensedTree {
        nodes: output,
        n_samples,
    }
}

/// Extract leaf cluster IDs from the condensed tree.
///
/// A leaf cluster is one that never appears as a parent of another cluster,
/// only of individual points. These are the candidates for final cluster
/// assignment.
///
/// ### Params
///
/// * `tree` - Condensed tree produced by `condense_tree()`
///
/// ### Returns
///
/// Sorted list of cluster node indices that are leaves in the condensed
/// hierarchy.
pub fn extract_leaves(tree: &CondensedTree<impl EvocFloat>) -> Vec<usize> {
    if tree.nodes.is_empty() {
        return Vec::new();
    }

    let max_id = tree
        .nodes
        .iter()
        .map(|n| n.parent.max(n.child))
        .max()
        .unwrap_or(0);

    let mut is_parent = vec![false; max_id + 1];
    for node in &tree.nodes {
        if node.child_size > 1 {
            is_parent[node.parent] = true;
        }
    }

    let mut leaves = Vec::new();
    for i in tree.n_samples..=max_id {
        if !is_parent[i] {
            // check it actually appears as a parent of at least one point
            let exists = tree.nodes.iter().any(|n| n.parent == i);
            if exists {
                leaves.push(i);
            }
        }
    }
    leaves
}

/// Assign each point to a cluster given a set of selected cluster IDs.
///
/// Points not falling under any selected cluster are labelled `-1` (noise).
///
/// ### Params
///
/// * `tree` - Condensed tree produced by `condense_tree()`.
/// * `clusters` - Selected cluster IDs, typically the output of
///   `extract_leaves()`
/// * `n_samples` - Number of original data points
///
/// ### Returns
///
/// Label vector of length `n_samples`; entry `i` is the cluster index of point
/// `i`, or `-1` if the point is noise
pub fn get_cluster_label_vector<T>(
    tree: &CondensedTree<T>,
    clusters: &[usize],
    n_samples: usize,
) -> Vec<i64>
where
    T: EvocFloat,
{
    if clusters.is_empty() {
        return vec![-1i64; n_samples];
    }

    let max_id = tree
        .nodes
        .iter()
        .map(|n| n.parent.max(n.child))
        .max()
        .unwrap_or(0);

    let root = tree
        .nodes
        .iter()
        .map(|n| n.parent)
        .min()
        .unwrap_or(n_samples);

    let mut ds = DisjointSet::new(max_id + 1);
    let cluster_set: std::collections::HashSet<usize> = clusters.iter().copied().collect();

    // merge everything that isn't a selected cluster boundary
    for node in &tree.nodes {
        if !cluster_set.contains(&node.child) {
            ds.union(node.parent, node.child);
        }
    }

    // build label map: sorted cluster ID -> label index
    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort();
    let label_map: std::collections::HashMap<usize, i64> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as i64))
        .collect();

    let mut labels = vec![-1i64; n_samples];
    for i in 0..n_samples {
        let cluster = ds.find(i);
        if cluster <= root {
            labels[i] = -1;
        } else if let Some(&label) = label_map.get(&cluster) {
            labels[i] = label;
        } else {
            labels[i] = -1;
        }
    }

    labels
}

/// Compute the maximum lambda (death lambda) for each selected cluster.
///
/// The death lambda is the largest lambda at which any point falls out of
/// the cluster, i.e. the lambda at which the cluster effectively dissolves.
///
/// ### Params
///
/// * `tree` - Condensed tree produced by `condense_tree()`
/// * `clusters` - Selected cluster IDs
///
/// ### Returns
///
/// Map from cluster ID to its maximum lambda value
fn max_lambdas<T>(
    tree: &CondensedTree<T>,
    clusters: &[usize],
) -> std::collections::HashMap<usize, T>
where
    T: EvocFloat,
{
    let cluster_set: std::collections::HashSet<usize> = clusters.iter().copied().collect();
    let mut result: std::collections::HashMap<usize, T> =
        clusters.iter().map(|&c| (c, T::zero())).collect();

    for node in &tree.nodes {
        if node.child_size == 1 && cluster_set.contains(&node.parent) {
            let entry = result.get_mut(&node.parent).unwrap();
            if node.lambda_val > *entry {
                *entry = node.lambda_val;
            }
        }
    }
    result
}

/// Compute membership strength for each point relative to its assigned cluster.
///
/// Strength is the ratio of the point's lambda to the cluster's death lambda,
/// clamped to `[0, 1]`. A strength of `1` means the point persisted until the
/// cluster dissolved; lower values indicate earlier departure.
///
/// ### Params
///
/// * `tree` - Condensed tree produced by `condense_tree()`
/// * `clusters` - Selected cluster IDs, typically the output of
///   `extract_leaves()`
/// * `labels` - Cluster label per point from `get_cluster_label_vector()`
///
/// ### Returns
///
/// Membership strength vector of length `labels.len()`; noise points (label `-1`)
/// retain a strength of `0`
pub fn get_point_membership_strengths<T>(
    tree: &CondensedTree<T>,
    clusters: &[usize],
    labels: &[i64],
) -> Vec<T>
where
    T: EvocFloat,
{
    let n = labels.len();
    let mut result = vec![T::zero(); n];
    let deaths = max_lambdas(tree, clusters);

    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort();

    // index -> cluster_id
    let index_to_cluster: std::collections::HashMap<i64, usize> = sorted_clusters
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as i64, c))
        .collect();

    let root = tree.nodes.iter().map(|n| n.parent).min().unwrap_or(n);

    for node in &tree.nodes {
        let point = node.child;
        if point >= root || point >= n {
            continue;
        }
        if labels[point] < 0 {
            continue;
        }

        let cluster_id = match index_to_cluster.get(&labels[point]) {
            Some(&c) => c,
            None => continue,
        };

        let max_lambda = match deaths.get(&cluster_id) {
            Some(&ml) => ml,
            None => continue,
        };

        if max_lambda == T::zero() || !node.lambda_val.is_finite() {
            result[point] = T::one();
        } else {
            let lv = if node.lambda_val < max_lambda {
                node.lambda_val
            } else {
                max_lambda
            };
            result[point] = lv / max_lambda;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::linkage::{LinkageRow, mst_to_linkage_tree};
    use crate::clustering::mst::build_mst;

    fn make_two_cluster_tree() -> (Vec<LinkageRow<f64>>, usize) {
        // two tight clusters: {0,1,2} near origin, {3,4,5} far away
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let mut mst = build_mst(&data, 1);
        let n = data.len();
        let linkage = mst_to_linkage_tree(&mut mst, n);
        (linkage, n)
    }

    #[test]
    fn test_condense_two_clusters() {
        let (linkage, n) = make_two_cluster_tree();
        let ct = condense_tree(&linkage, n, 2);

        let leaves = extract_leaves(&ct);
        assert_eq!(leaves.len(), 2, "expected 2 leaf clusters");
    }

    #[test]
    fn test_condense_min_cluster_size_too_large() {
        let (linkage, n) = make_two_cluster_tree();
        // min_cluster_size larger than either cluster
        let ct = condense_tree(&linkage, n, n);

        let leaves = extract_leaves(&ct);
        // everything collapses into root, no leaf clusters with children
        assert!(leaves.len() <= 1);
    }

    #[test]
    fn test_label_vector_two_clusters() {
        let (linkage, n) = make_two_cluster_tree();
        let ct = condense_tree(&linkage, n, 2);
        let leaves = extract_leaves(&ct);
        let labels = get_cluster_label_vector(&ct, &leaves, n);

        assert_eq!(labels.len(), n);

        // points 0,1,2 should share a label; 3,4,5 should share another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);

        // the two groups should differ
        assert_ne!(labels[0], labels[3]);

        // no noise expected
        assert!(labels.iter().all(|&l| l >= 0));
    }

    #[test]
    fn test_membership_strengths_range() {
        let (linkage, n) = make_two_cluster_tree();
        let ct = condense_tree(&linkage, n, 2);
        let leaves = extract_leaves(&ct);
        let labels = get_cluster_label_vector(&ct, &leaves, n);
        let strengths = get_point_membership_strengths(&ct, &leaves, &labels);

        assert_eq!(strengths.len(), n);
        for (i, &s) in strengths.iter().enumerate() {
            if labels[i] >= 0 {
                assert!(
                    (0.0..=1.0).contains(&s),
                    "strength {} out of range: {}",
                    i,
                    s
                );
            }
        }
    }

    #[test]
    fn test_condense_empty_linkage() {
        let ct = condense_tree::<f64>(&[], 0, 2);
        assert!(ct.nodes.is_empty());
        assert!(extract_leaves(&ct).is_empty());
    }

    #[test]
    fn test_labels_with_noise() {
        // 5 points: tight pair + 3 scattered
        let data = vec![
            vec![0.0, 0.0],
            vec![0.05, 0.0],
            vec![5.0, 5.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        let mut mst = build_mst(&data, 1);
        let n = data.len();
        let linkage = mst_to_linkage_tree(&mut mst, n);
        let ct = condense_tree(&linkage, n, 2);
        let leaves = extract_leaves(&ct);
        let labels = get_cluster_label_vector(&ct, &leaves, n);

        // the tight pair should be in the same cluster
        if labels[0] >= 0 && labels[1] >= 0 {
            assert_eq!(labels[0], labels[1]);
        }
    }
}
