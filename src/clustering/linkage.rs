//! Single-linkage dendrogram construction from a minimum spanning tree.
//!
//! Converts the MST produced by [`crate::clustering::mst`] into a linkage
//! matrix in scipy's format: each row describes one merge, identifying the
//! two nodes joined, the merge distance, and the resulting cluster size.
//! Internal nodes are labelled `n_samples`, `n_samples + 1`, … in merge
//! order. Leaf nodes retain their original point indices.

use crate::clustering::mst::MstEdge;
use crate::prelude::*;

/// One row of the scipy-style linkage matrix, describing a single merge.
///
/// Node indices below `n_samples` are leaves (original points); indices
/// `n_samples` and above are internal nodes created during merging.
#[derive(Clone, Debug)]
pub struct LinkageRow<T> {
    /// Index of the larger (or equal) of the two merged nodes
    pub left: usize,
    /// Index of the smaller of the two merged nodes
    pub right: usize,
    /// Mutual reachability distance at which the merge occurs
    pub distance: T,
    /// Total number of original points in the merged cluster
    pub size: usize,
}

/// Convert an MST into a single-linkage dendrogram (scipy-style linkage
/// matrix).
///
/// ### Params
///
/// * `mst` - MST edges, sorted in place by ascending weight
/// * `n_samples` - Number of original data points (leaf nodes)
///
/// ### Returns
///
/// `Vec<LinkageRow<T>>` of length `n_samples - 1`
pub fn mst_to_linkage_tree<T>(mst: &mut [MstEdge<T>], n_samples: usize) -> Vec<LinkageRow<T>>
where
    T: EvocFloat,
{
    mst.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap());

    let n_merges = mst.len();
    let total_nodes = 2 * n_samples - 1;

    // parent/size tracking for relabelling into scipy convention
    let mut parent = vec![usize::MAX; total_nodes];
    let mut size = vec![0usize; total_nodes];
    for i in 0..n_samples {
        size[i] = 1;
    }

    let mut linkage = Vec::with_capacity(n_merges);
    let mut next_label = n_samples;

    for edge in mst.iter() {
        let mut left = edge.u;
        let mut right = edge.v;

        // chase up to current root label
        while parent[left] != usize::MAX {
            left = parent[left];
        }
        while parent[right] != usize::MAX {
            right = parent[right];
        }

        let new_size = size[left] + size[right];

        // convention: larger index first (matches Python)
        if left < right {
            std::mem::swap(&mut left, &mut right);
        }

        linkage.push(LinkageRow {
            left,
            right,
            distance: edge.weight,
            size: new_size,
        });

        parent[left] = next_label;
        parent[right] = next_label;
        size[next_label] = new_size;
        next_label += 1;
    }

    linkage
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::mst::MstEdge;

    #[test]
    fn test_linkage_basic() {
        // 3 points: 0--1 (dist 1), 1--2 (dist 2)
        let mut mst = vec![
            MstEdge {
                u: 0,
                v: 1,
                weight: 1.0,
            },
            MstEdge {
                u: 1,
                v: 2,
                weight: 2.0,
            },
        ];

        let linkage = mst_to_linkage_tree(&mut mst, 3);
        assert_eq!(linkage.len(), 2);

        // first merge: 0 and 1 at distance 1, size 2
        assert_eq!(linkage[0].distance, 1.0);
        assert_eq!(linkage[0].size, 2);

        // second merge: cluster(0,1) and 2 at distance 2, size 3
        assert_eq!(linkage[1].distance, 2.0);
        assert_eq!(linkage[1].size, 3);
    }

    #[test]
    fn test_linkage_unsorted_input() {
        // feed edges in wrong order -- should still work
        let mut mst = vec![
            MstEdge {
                u: 1,
                v: 2,
                weight: 5.0,
            },
            MstEdge {
                u: 0,
                v: 1,
                weight: 1.0,
            },
        ];

        let linkage = mst_to_linkage_tree(&mut mst, 3);
        assert_eq!(linkage[0].distance, 1.0);
        assert_eq!(linkage[1].distance, 5.0);
    }

    #[test]
    fn test_linkage_sizes() {
        // 4 points forming a chain: 0-1(1), 1-2(2), 2-3(3)
        let mut mst = vec![
            MstEdge {
                u: 0,
                v: 1,
                weight: 1.0,
            },
            MstEdge {
                u: 1,
                v: 2,
                weight: 2.0,
            },
            MstEdge {
                u: 2,
                v: 3,
                weight: 3.0,
            },
        ];

        let linkage = mst_to_linkage_tree(&mut mst, 4);
        assert_eq!(linkage.len(), 3);
        assert_eq!(linkage[0].size, 2);
        assert_eq!(linkage[1].size, 3);
        assert_eq!(linkage[2].size, 4);
    }

    #[test]
    fn test_linkage_monotonic_distances() {
        let mut mst = vec![
            MstEdge {
                u: 0,
                v: 1,
                weight: 3.0,
            },
            MstEdge {
                u: 2,
                v: 3,
                weight: 1.0,
            },
            MstEdge {
                u: 1,
                v: 2,
                weight: 5.0,
            },
        ];

        let linkage = mst_to_linkage_tree(&mut mst, 4);
        for i in 1..linkage.len() {
            assert!(linkage[i].distance >= linkage[i - 1].distance);
        }
    }

    #[test]
    fn test_linkage_two_simultaneous_merges() {
        // two separate pairs merging at the same distance, then joined
        let mut mst = vec![
            MstEdge {
                u: 0,
                v: 1,
                weight: 1.0,
            },
            MstEdge {
                u: 2,
                v: 3,
                weight: 1.0,
            },
            MstEdge {
                u: 0,
                v: 2,
                weight: 10.0,
            },
        ];

        let linkage = mst_to_linkage_tree(&mut mst, 4);
        assert_eq!(linkage.len(), 3);
        assert_eq!(linkage[0].size, 2);
        assert_eq!(linkage[1].size, 2);
        assert_eq!(linkage[2].size, 4);
    }
}
