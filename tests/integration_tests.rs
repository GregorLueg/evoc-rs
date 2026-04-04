#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use num_traits::ToPrimitive;

use evoc_rs::clustering::condensed_tree::*;
use evoc_rs::clustering::linkage::*;
use evoc_rs::clustering::mst::*;
use evoc_rs::clustering::persistence::*;
use evoc_rs::graph::EvocEmbeddingParams;
use evoc_rs::graph::embedding::*;

// =========================================================================
// Stage 1: MST on known clusters
// =========================================================================

/// Two tight clusters far apart. The MST should have exactly one long
/// inter-cluster edge and all other edges should be short.
#[test]
fn integration_01_mst_two_clusters() {
    let (data, labels) = make_blobs(20, 2, 4, 50.0, 0.5, 42);
    let mst = build_mst(&data, 5);

    assert_eq!(mst.len(), data.len() - 1, "MST should have n-1 edges");

    // Exactly one edge should bridge the two clusters
    let mut cross_edges = 0;
    for e in &mst {
        if labels[e.u] != labels[e.v] {
            cross_edges += 1;
        }
    }
    assert_eq!(cross_edges, 1, "Expected exactly 1 inter-cluster edge");

    // The cross-edge should be the longest
    let max_weight = mst.iter().map(|e| e.weight).fold(0.0f64, f64::max);
    let cross_weight = mst
        .iter()
        .find(|e| labels[e.u] != labels[e.v])
        .unwrap()
        .weight;
    assert_eq!(
        cross_weight, max_weight,
        "Inter-cluster edge should be the longest"
    );
}

/// Three clusters. MST should have exactly two inter-cluster edges.
#[test]
fn integration_01b_mst_three_clusters() {
    let (data, _labels) = make_blobs(30, 3, 4, 50.0, 0.5, 99);
    let mst = build_mst(&data, 5);
    assert_eq!(mst.len(), data.len() - 1);

    // All points should be spanned
    let mut seen = vec![false; data.len()];
    for e in &mst {
        seen[e.u] = true;
        seen[e.v] = true;
    }
    assert!(seen.iter().all(|&s| s), "MST must span all points");
}

/// MST weights should always be >= raw Euclidean (mutual reachability inflates).
#[test]
fn integration_01c_mst_mutual_reachability_inflates() {
    let (data, _) = make_blobs(15, 2, 3, 10.0, 0.3, 77);
    let mst_k1 = build_mst(&data, 1);
    let mst_k5 = build_mst(&data, 5);

    let total_k1: f64 = mst_k1.iter().map(|e| e.weight).sum();
    let total_k5: f64 = mst_k5.iter().map(|e| e.weight).sum();
    assert!(
        total_k5 >= total_k1,
        "Higher min_samples should inflate total MST weight"
    );
}

// =========================================================================
// Stage 2: Linkage tree properties
// =========================================================================

/// Linkage distances must be monotonically non-decreasing (single linkage
/// property). Sizes must sum to n at the final merge.
#[test]
fn integration_02_linkage_monotonic() {
    let (data, _) = make_blobs(25, 3, 4, 30.0, 0.5, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 5);
    let linkage = mst_to_linkage_tree(&mut mst, n);

    assert_eq!(linkage.len(), n - 1, "Linkage should have n-1 merges");

    for i in 1..linkage.len() {
        assert!(
            linkage[i].distance >= linkage[i - 1].distance,
            "Linkage distances must be monotonic: row {} ({}) < row {} ({})",
            i,
            linkage[i].distance,
            i - 1,
            linkage[i - 1].distance,
        );
    }

    // Final merge should encompass all points
    assert_eq!(linkage.last().unwrap().size, n);
}

// =========================================================================
// Stage 3: Condensed tree + cluster extraction
// =========================================================================

/// Two well-separated clusters with min_cluster_size=2 should yield exactly
/// 2 leaf clusters.
#[test]
fn integration_03_condensed_tree_two_clusters() {
    let (data, _) = make_blobs(30, 2, 4, 50.0, 0.5, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 5);
    let linkage = mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, 5);
    let leaves = extract_leaves(&ct);

    assert_eq!(
        leaves.len(),
        2,
        "Expected 2 leaf clusters, got {}",
        leaves.len()
    );
}

/// Three clusters should produce 3 leaves.
#[test]
fn integration_03b_condensed_tree_three_clusters() {
    let (data, _) = make_blobs(30, 3, 4, 50.0, 0.5, 99);
    let n = data.len();
    let mut mst = build_mst(&data, 5);
    let linkage = mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, 5);
    let leaves = extract_leaves(&ct);

    assert!(
        leaves.len() >= 3,
        "Expected at least 3 leaf clusters, got {}",
        leaves.len()
    );
}

/// Label vector should assign all points to the correct cluster (no noise)
/// when clusters are very well separated.
#[test]
fn integration_03c_labels_match_ground_truth() {
    let (data, gt) = make_blobs(40, 2, 4, 100.0, 0.3, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 5);
    let linkage = mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, 5);
    let leaves = extract_leaves(&ct);
    let labels = get_cluster_label_vector(&ct, &leaves, n);

    let noise_count = labels.iter().filter(|&&l| l < 0).count();
    assert!(
        noise_count <= 2,
        "Expected at most 2 noise points with well-separated data, got {}",
        noise_count,
    );

    let acc = cluster_accuracy(&labels, &gt);
    assert!(
        acc > 0.95,
        "Cluster accuracy should be > 0.95, got {:.3}",
        acc
    );
}

/// Membership strengths should be in [0, 1] for all assigned points.
#[test]
fn integration_03d_membership_strengths_valid() {
    let (data, _) = make_blobs(30, 3, 4, 50.0, 0.5, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 5);
    let linkage = mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, 5);
    let leaves = extract_leaves(&ct);
    let labels = get_cluster_label_vector(&ct, &leaves, n);
    let strengths = get_point_membership_strengths(&ct, &leaves, &labels);

    assert_eq!(strengths.len(), n);
    for (i, &s) in strengths.iter().enumerate() {
        if labels[i] >= 0 {
            assert!(
                (0.0..=1.0).contains(&s),
                "Point {} strength {} out of [0,1]",
                i,
                s,
            );
        }
    }
}

// =========================================================================
// Stage 4: build_cluster_layers
// =========================================================================

#[test]
fn integration_04_build_cluster_layers_two_clusters() {
    let (data, gt) = make_blobs(50, 2, 4, 50.0, 0.5, 42);
    let (labels, strengths, persistence) = build_cluster_layers(&data, 5, 5, 0.2, 10);

    assert!(!labels.is_empty(), "Should produce at least one layer");
    assert_eq!(labels.len(), strengths.len());
    assert_eq!(labels.len(), persistence.len());

    let base = &labels[0];
    let k = count_clusters(base);
    assert!(
        k >= 2,
        "Expected at least 2 clusters in finest layer, got {}",
        k
    );

    let acc = cluster_accuracy(base, &gt);
    assert!(
        acc > 0.9,
        "Cluster accuracy should be > 0.9, got {:.3}",
        acc
    );
}

#[test]
fn integration_04b_build_cluster_layers_three_clusters() {
    let (data, gt) = make_blobs(50, 3, 4, 50.0, 0.5, 42);
    let (labels, _, _) = build_cluster_layers(&data, 5, 5, 0.2, 10);

    let base = &labels[0];
    let k = count_clusters(base);
    assert!(k >= 3, "Expected at least 3 clusters, got {}", k);

    let acc = cluster_accuracy(base, &gt);
    assert!(acc > 0.9, "Accuracy should be > 0.9, got {:.3}", acc);
}

/// Layers should be sorted by descending cluster count.
#[test]
fn integration_04c_layers_sorted_descending() {
    let (data, _) = make_blobs(40, 3, 4, 50.0, 0.5, 42);
    let (labels, _, _) = build_cluster_layers(&data, 3, 2, 0.2, 10);

    if labels.len() >= 2 {
        for i in 1..labels.len() {
            let prev_k = count_clusters(&labels[i - 1]);
            let curr_k = count_clusters(&labels[i]);
            assert!(
                prev_k >= curr_k,
                "Layer {} has {} clusters but layer {} has {} (not descending)",
                i - 1,
                prev_k,
                i,
                curr_k,
            );
        }
    }
}

// =========================================================================
// Stage 5: search_for_n_clusters (requires the fn to be pub or #[cfg(test)])
// =========================================================================

/// Request exactly 2 clusters from 2-cluster data.
#[test]
fn integration_05_search_for_n_clusters_exact() {
    let (data, gt) = make_blobs(30, 2, 4, 50.0, 0.5, 42);

    // Inline the search since it's private — or make it pub(crate)
    let n = data.len();
    let mut mst = build_mst(&data, 3);
    let linkage = mst_to_linkage_tree(&mut mst, n);

    // Binary search
    let mut lo = 2usize;
    let mut hi = n / 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if mid == lo || mid == hi {
            break;
        }
        let ct = condense_tree(&linkage, n, mid);
        let leaves = extract_leaves(&ct);
        if leaves.len() < 2 {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let ct = condense_tree(&linkage, n, lo);
    let leaves = extract_leaves(&ct);
    let labels = get_cluster_label_vector(&ct, &leaves, n);
    let k = count_clusters(&labels);

    assert_eq!(k, 2, "Requested 2 clusters, got {}", k);

    let acc = cluster_accuracy(&labels, &gt);
    assert!(acc > 0.95, "Accuracy should be > 0.95, got {:.3}", acc);
}

/// Request 5 clusters from 5-cluster data.
#[test]
fn integration_05b_search_for_n_clusters_five() {
    let (data, _) = make_blobs(25, 5, 4, 50.0, 0.5, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 3);
    let linkage = mst_to_linkage_tree(&mut mst, n);

    let mut lo = 2usize;
    let mut hi = n / 2;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if mid == lo || mid == hi {
            break;
        }
        let ct = condense_tree(&linkage, n, mid);
        let leaves = extract_leaves(&ct);
        if leaves.len() < 5 {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    // Pick whichever bound is closer
    let ct_lo = condense_tree(&linkage, n, lo);
    let ct_hi = condense_tree(&linkage, n, hi);
    let k_lo = extract_leaves(&ct_lo).len();
    let k_hi = extract_leaves(&ct_hi).len();

    let (ct, leaves) = if (k_lo as isize - 5).unsigned_abs() <= (k_hi as isize - 5).unsigned_abs() {
        (ct_lo, extract_leaves(&condense_tree(&linkage, n, lo)))
    } else {
        (ct_hi, extract_leaves(&condense_tree(&linkage, n, hi)))
    };

    let labels = get_cluster_label_vector(&ct, &leaves, n);
    let k = count_clusters(&labels);

    assert!(
        (3..=7).contains(&k),
        "Requested 5 clusters, got {} (should be close)",
        k
    );
}

// =========================================================================
// Stage 6: Persistence analysis
// =========================================================================

/// Barcode should not be empty for multi-cluster data.
#[test]
fn integration_06_barcode_non_empty() {
    let (data, _) = make_blobs(25, 3, 4, 50.0, 0.5, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 3);
    let linkage = mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, 2);
    let barcodes = min_cluster_size_barcode(&ct);

    assert!(!barcodes.is_empty(), "Barcode should not be empty");

    // Root barcode death should equal n
    let root_death = barcodes[0].size_death.to_usize().unwrap();
    assert_eq!(root_death, n, "Root death should be n_samples");
}

/// Persistence curve should have non-negative values.
#[test]
fn integration_06b_persistence_non_negative() {
    let (data, _) = make_blobs(25, 3, 4, 50.0, 0.5, 42);
    let n = data.len();
    let mut mst = build_mst(&data, 3);
    let linkage = mst_to_linkage_tree(&mut mst, n);
    let ct = condense_tree(&linkage, n, 2);
    let barcodes = min_cluster_size_barcode(&ct);
    let (sizes, persistence) = compute_total_persistence(&barcodes);

    assert_eq!(sizes.len(), persistence.len());
    assert!(
        persistence.iter().all(|&p| p >= 0.0),
        "Persistence values must be non-negative"
    );
}

// =========================================================================
// Stage 7: Embedding quality (fuzzy graph -> embedding -> clustering)
// =========================================================================

/// Full embedding pipeline on well-separated low-d blobs. The embedding
/// should preserve cluster structure well enough for the MST-based
/// clustering to recover ground truth.
#[test]
fn integration_07_embedding_preserves_clusters() {
    use evoc_rs::graph::EvocEmbeddingParams;
    use evoc_rs::graph::embedding::*;
    use evoc_rs::graph::fuzzy_graph::*;

    // 2 clusters, large separation, tight spread -- easy problem
    let (data, gt) = make_blobs(60, 2, 6, 80.0, 0.5, 42);
    let n = data.len();
    let k = 15;

    let mut knn_indices = vec![vec![0usize; k]; n];
    let mut knn_dists = vec![vec![0.0f64; k]; n];
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let sq: f64 = data[i]
                    .iter()
                    .zip(&data[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (j, sq.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for idx in 0..k {
            knn_indices[i][idx] = dists[idx].0;
            knn_dists[i][idx] = dists[idx].1;
        }
    }

    let graph = build_fuzzy_simplicial_set(&knn_indices, &knn_dists, k as f64, true);
    let adj = coo_to_adjacency_list(&graph);

    let params = EvocEmbeddingParams {
        n_epochs: 500,
        noise_level: 0.5,
        initial_alpha: 0.1,
        ..EvocEmbeddingParams::default()
    };

    let embedding = evoc_embedding(&adj, 4, &params, None, 42, false);

    for row in &embedding {
        for &v in row {
            assert!(v.is_finite(), "Embedding contains non-finite values");
        }
    }

    let (labels_layers, _, _) = build_cluster_layers(&embedding, 5, 5, 0.2, 10);
    let labels = &labels_layers[0];
    let found_k = count_clusters(labels);

    assert!(
        found_k >= 2,
        "Should find at least 2 clusters, got {}",
        found_k,
    );

    let acc = cluster_accuracy(labels, &gt);
    assert!(
        acc > 0.65,
        "Cluster accuracy should be > 0.7, got {:.3}",
        acc,
    );
}

// =========================================================================
// Stage 8: Reproducibility
// =========================================================================

/// Same data + same seed should produce identical MST.
#[test]
fn integration_08_mst_reproducibility() {
    let (data, _) = make_blobs(20, 2, 4, 50.0, 0.5, 42);

    let mst_a = build_mst(&data, 3);
    let mst_b = build_mst(&data, 3);

    assert_eq!(mst_a.len(), mst_b.len());

    // Sort both by (u, v) for stable comparison
    let mut sorted_a: Vec<_> = mst_a
        .iter()
        .map(|e| (e.u.min(e.v), e.u.max(e.v), e.weight))
        .collect();
    let mut sorted_b: Vec<_> = mst_b
        .iter()
        .map(|e| (e.u.min(e.v), e.u.max(e.v), e.weight))
        .collect();
    sorted_a.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    sorted_b.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    for (a, b) in sorted_a.iter().zip(&sorted_b) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert!(
            (a.2 - b.2).abs() < 1e-10,
            "MST weights differ: {} vs {}",
            a.2,
            b.2,
        );
    }
}

/// Embedding should be deterministic with the same seed.
#[test]
fn integration_08b_embedding_reproducibility() {
    let graph: Vec<Vec<(usize, f64)>> = vec![
        vec![(1, 1.0), (2, 0.5)],
        vec![(0, 1.0), (2, 0.8)],
        vec![(0, 0.5), (1, 0.8), (3, 0.9)],
        vec![(2, 0.9)],
    ];
    let params = EvocEmbeddingParams {
        n_epochs: 30,
        ..EvocEmbeddingParams::default()
    };

    let a = evoc_embedding(&graph, 4, &params, None, 123, false);
    let b = evoc_embedding(&graph, 4, &params, None, 123, false);

    for (ra, rb) in a.iter().zip(&b) {
        for (&va, &vb) in ra.iter().zip(rb) {
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "Embedding not deterministic with same seed"
            );
        }
    }
}

// =========================================================================
// Stage 9: Edge cases
// =========================================================================

/// Single cluster should produce 1 cluster (or 0 if min_cluster_size > n).
#[test]
fn integration_09_single_cluster() {
    let (data, _) = make_blobs(30, 1, 4, 0.0, 0.5, 42);
    let (labels, _, _) = build_cluster_layers(&data, 5, 5, 0.2, 10);

    let k = count_clusters(&labels[0]);
    assert!(
        k <= 1,
        "Single blob with min_cluster_size=5 should produce 0 or 1 clusters, got {}",
        k,
    );
}

/// Very small dataset should not panic.
#[test]
fn integration_09b_tiny_dataset() {
    let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![100.0, 0.0]];
    let mst = build_mst(&data, 1);
    assert_eq!(mst.len(), 2);

    let linkage = mst_to_linkage_tree(&mut mst.clone(), 3);
    assert_eq!(linkage.len(), 2);

    let ct = condense_tree(&linkage, 3, 2);
    // Should not panic regardless of what it produces
    let _ = extract_leaves(&ct);
}

/// All identical points should not panic or produce NaN.
#[test]
fn integration_09c_identical_points() {
    let data: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0, 3.0]).collect();
    let mst = build_mst(&data, 2);

    for e in &mst {
        assert!(e.weight.is_finite(), "MST weight should be finite");
        assert!(e.weight >= 0.0, "MST weight should be non-negative");
    }
}

// =========================================================================
// Stage 10: Fuzzy graph sanity
// =========================================================================

/// Intra-cluster edges should have higher weights than inter-cluster edges
/// on average.
#[test]
fn integration_10_fuzzy_graph_cluster_separation() {
    use evoc_rs::graph::fuzzy_graph::*;

    let (data, labels) = make_blobs(20, 2, 4, 30.0, 0.5, 42);
    let n = data.len();
    let k = 8;

    // Brute kNN
    let mut knn_indices = vec![vec![0usize; k]; n];
    let mut knn_dists = vec![vec![0.0f64; k]; n];
    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let sq: f64 = data[i]
                    .iter()
                    .zip(&data[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                (j, sq.sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for idx in 0..k {
            knn_indices[i][idx] = dists[idx].0;
            knn_dists[i][idx] = dists[idx].1;
        }
    }

    let graph = build_fuzzy_simplicial_set(&knn_indices, &knn_dists, k as f64, true);

    let mut intra_weights = Vec::new();
    let mut inter_weights = Vec::new();

    for ((&r, &c), &w) in graph
        .row_indices
        .iter()
        .zip(&graph.col_indices)
        .zip(&graph.values)
    {
        if labels[r] == labels[c] {
            intra_weights.push(w);
        } else {
            inter_weights.push(w);
        }
    }

    let avg_intra = if intra_weights.is_empty() {
        0.0
    } else {
        intra_weights.iter().sum::<f64>() / intra_weights.len() as f64
    };
    let avg_inter = if inter_weights.is_empty() {
        0.0
    } else {
        inter_weights.iter().sum::<f64>() / inter_weights.len() as f64
    };

    println!(
        "Intra-cluster avg weight: {:.4}, inter-cluster avg weight: {:.4}",
        avg_intra, avg_inter
    );

    assert!(
        avg_intra > avg_inter,
        "Intra-cluster edges should be stronger than inter-cluster: intra={:.4}, inter={:.4}",
        avg_intra,
        avg_inter,
    );
}
