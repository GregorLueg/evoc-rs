#![cfg(mlx_available)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

mod commons;
use commons::*;

use ann_search_rs::synthetic::generate_clustered_data;

use evoc_rs::nearest_neighbours::nearest_neighbour_cpu::run_ann_search;
use evoc_rs::prelude::*;
use evoc_rs::{EvocParams, evoc};

/// Same fixture as the GPU accuracy tests.
const ACC_FIXTURE: (usize, usize, usize, u64) = (600, 32, 4, 1);

/// The MLX backends, as `ann_type` strings.
const MLX_BACKENDS: [&str; 3] = ["exhaustive_mlx", "ivf_mlx", "nndescent_mlx"];

/// Every MLX backend should recover the clusters end to end.
#[test]
fn mlx_integration_01_clusters_all_backends() {
    let (n, dim, nc, seed) = ACC_FIXTURE;
    let (mat, gt) = generate_clustered_data::<f32>(n, dim, nc, seed);

    let params = EvocParams::<f32>::default();
    let nn_params = NearestNeighbourParamsEvoc::<f32> {
        n_list: Some(5),
        n_probes: Some(5),
        ..NearestNeighbourParamsEvoc::default()
    };

    for ann_type in MLX_BACKENDS {
        let result = evoc(
            mat.as_ref(),
            ann_type.to_string(),
            None,
            &params,
            &nn_params,
            42,
            0,
        )
        .unwrap();

        let acc = cluster_accuracy(result.best_labels(), &gt);
        println!(
            "Accuracy ({}): {:.3}, k = {}",
            ann_type,
            acc,
            result.n_clusters()
        );
        assert!(acc > 0.95, "{} accuracy {:.3} too low", ann_type, acc);
    }
}

/// Every backend returns exactly `k` neighbours per point, none of them self,
/// including an `f64` input.
#[test]
fn mlx_integration_02_knn_shape_no_self() {
    let (n, dim, nc, seed) = ACC_FIXTURE;
    let (mat, _) = generate_clustered_data::<f64>(n, dim, nc, seed);
    let k = 15;

    let nn_params = NearestNeighbourParamsEvoc::<f64>::default();

    for ann_type in MLX_BACKENDS {
        let (idx, dist) =
            run_ann_search(mat.as_ref(), k, ann_type.to_string(), &nn_params, 42, 0).unwrap();

        assert_eq!(idx.len(), n);
        for i in 0..n {
            assert_eq!(idx[i].len(), k, "{}: row {} has wrong length", ann_type, i);
            assert_eq!(dist[i].len(), k);
            assert!(
                !idx[i].contains(&i),
                "{}: row {} contains self",
                ann_type,
                i
            );
        }
    }
}

/// Exhaustive MLX agrees with the exhaustive CPU search on neighbours and on
/// the (true, not squared) Euclidean distance scale.
#[test]
fn mlx_integration_03_exhaustive_matches_cpu() {
    let (n, dim, nc, seed) = ACC_FIXTURE;
    let (mat, _) = generate_clustered_data::<f32>(n, dim, nc, seed);
    let k = 15;
    let nn_params = NearestNeighbourParamsEvoc::<f32>::default();

    let (idx_cpu, dist_cpu) =
        run_ann_search(mat.as_ref(), k, "exhaustive".to_string(), &nn_params, 42, 0).unwrap();
    let (idx_mlx, dist_mlx) = run_ann_search(
        mat.as_ref(),
        k,
        "exhaustive_mlx".to_string(),
        &nn_params,
        42,
        0,
    )
    .unwrap();

    let mut hits = 0;
    for i in 0..n {
        hits += idx_mlx[i].iter().filter(|j| idx_cpu[i].contains(j)).count();
        // The k-th distance is robust to ties swapping neighbours.
        let (a, b) = (dist_cpu[i][k - 1], dist_mlx[i][k - 1]);
        assert!(
            (a - b).abs() <= 1e-3 * a.max(1.0),
            "row {}: CPU {} vs MLX {}",
            i,
            a,
            b
        );
    }
    let recall = hits as f64 / (n * k) as f64;
    println!("Exhaustive MLX recall vs CPU: {:.4}", recall);
    assert!(recall > 0.99, "recall {:.4} too low", recall);
}
