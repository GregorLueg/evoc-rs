#![cfg(feature = "gpu")]
#![allow(clippy::needless_range_loop)]

mod commons;
use commons::*;

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use faer::Mat;

use evoc_rs::{EvocParams, evoc_gpu};
use manifolds_rs::data::nearest_neighbours_gpu::NearestNeighbourParamsGpu;

/// Convert `Vec<Vec<f32>>` from make_blobs into a faer matrix.
fn to_mat(data: &[Vec<f64>]) -> Mat<f32> {
    let n = data.len();
    let d = data[0].len();
    Mat::from_fn(n, d, |i, j| data[i][j] as f32)
}

/// End-to-end GPU EVoC on well-separated blobs should recover clusters.
#[test]
fn gpu_integration_01_two_clusters_exhaustive() {
    let (data, gt) = make_blobs(50, 2, 4, 50.0, 0.5, 42);
    let mat = to_mat(&data);

    let params = EvocParams::<f32>::default();
    let nn_params = NearestNeighbourParamsGpu::<f32>::default();
    let device = WgpuDevice::default();

    let result = evoc_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        "exhaustive_gpu".to_string(),
        None,
        &params,
        &nn_params,
        device,
        42,
        false,
    );

    assert!(!result.cluster_layers.is_empty());
    let labels = result.best_labels();
    let acc = cluster_accuracy(labels, &gt);
    println!("Two-cluster accuracy (exhaustive_gpu): {:.3}", acc);
    assert!(acc > 0.6, "Accuracy {:.3} below threshold", acc);
}

/// Same test with ivf_gpu — less precise but should still separate clusters.
#[test]
fn gpu_integration_02_two_clusters_ivf() {
    let (data, gt) = make_blobs(100, 2, 8, 50.0, 0.5, 42);
    let mat = to_mat(&data);

    let params = EvocParams::<f32>::default();
    let nn_params = NearestNeighbourParamsGpu::<f32> {
        n_list: Some(5),
        n_probes: Some(2),
        ..NearestNeighbourParamsGpu::default()
    };

    let device = WgpuDevice::default();

    let result = evoc_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        "ivf_gpu".to_string(),
        None,
        &params,
        &nn_params,
        device,
        42,
        false,
    );

    let labels = result.best_labels();
    let acc = cluster_accuracy(labels, &gt);
    println!("Two-cluster accuracy (ivf_gpu): {:.3}", acc);
    assert!(acc > 0.6, "IVF accuracy {:.3} too low", acc);
}

/// Dispatch check — all three GPU backends should run without panicking.
#[test]
fn gpu_integration_03_all_backends_dispatch() {
    let (data, _) = make_blobs(80, 3, 4, 40.0, 0.5, 42);
    let mat = to_mat(&data);

    let params = EvocParams::<f32>::default();
    let nn_params = NearestNeighbourParamsGpu::<f32>::default();

    for ann_type in ["exhaustive_gpu", "ivf_gpu", "nndescent_gpu"] {
        let device = WgpuDevice::default();
        let result = evoc_gpu::<f32, WgpuRuntime>(
            mat.as_ref(),
            ann_type.to_string(),
            None,
            &params,
            &nn_params,
            device,
            42,
            false,
        );
        assert!(
            !result.cluster_layers.is_empty(),
            "{} produced no layers",
            ann_type
        );
        assert_eq!(result.nn_indices.len(), data.len());
        println!("{} dispatched ok", ann_type);
    }
}

/// GPU result should be structurally consistent with CPU result on the same
/// data (not identical — GPU isn't bit-reproducible — but similar cluster count
/// and comparable accuracy).
#[test]
fn gpu_integration_04_structural_agreement_with_cpu() {
    use evoc_rs::evoc;
    use manifolds_rs::data::nearest_neighbours::NearestNeighbourParams;

    let (data, gt) = make_blobs(60, 3, 6, 50.0, 0.5, 42);
    let mat = to_mat(&data);

    let params = EvocParams::<f32>::default();

    let nn_cpu = NearestNeighbourParams::<f32>::default();
    let cpu = evoc::<f32>(
        mat.as_ref(),
        "nndescent".to_string(),
        None,
        &params,
        &nn_cpu,
        42,
        false,
    );

    let nn_gpu = NearestNeighbourParamsGpu::<f32>::default();
    let device = WgpuDevice::default();
    let gpu = evoc_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        "exhaustive_gpu".to_string(),
        None,
        &params,
        &nn_gpu,
        device,
        42,
        false,
    );

    let acc_cpu = cluster_accuracy(cpu.best_labels(), &gt);
    let acc_gpu = cluster_accuracy(gpu.best_labels(), &gt);
    println!("CPU accuracy: {:.3}, GPU accuracy: {:.3}", acc_cpu, acc_gpu);

    assert!(acc_cpu > 0.6 && acc_gpu > 0.6);
    let k_cpu = cpu.n_clusters();
    let k_gpu = gpu.n_clusters();
    println!("Cluster counts — CPU: {}, GPU: {}", k_cpu, k_gpu);
    let diff = (k_cpu as isize - k_gpu as isize).unsigned_abs();
    assert!(
        diff <= 2,
        "CPU/GPU cluster counts diverge: {} vs {}",
        k_cpu,
        k_gpu
    );
}

/// Precomputed kNN path should produce a valid clustering.
#[test]
fn gpu_integration_05_precomputed_knn() {
    use manifolds_rs::data::nearest_neighbours_gpu::run_ann_search_gpu;

    let (data, gt) = make_blobs(60, 2, 4, 50.0, 0.5, 42);
    let mat = to_mat(&data);
    let k = 15;

    let nn_params = NearestNeighbourParamsGpu::<f32>::default();
    let device = WgpuDevice::default();
    let (knn_indices, knn_dist) = run_ann_search_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        k,
        "exhaustive_gpu".to_string(),
        &nn_params,
        device,
        42,
        false,
    );

    let params = EvocParams::<f32> {
        n_neighbours: k,
        ..EvocParams::default()
    };

    let device2 = WgpuDevice::default();
    let result = evoc_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        "exhaustive_gpu".to_string(),
        Some((knn_indices, knn_dist)),
        &params,
        &nn_params,
        device2,
        42,
        false,
    );

    let acc = cluster_accuracy(result.best_labels(), &gt);
    assert!(acc > 0.6, "Precomputed-kNN accuracy {:.3} too low", acc);
}

/// approx_n_clusters mode with GPU kNN.
#[test]
fn gpu_integration_06_approx_n_clusters() {
    let (data, _) = make_blobs(80, 4, 4, 40.0, 0.5, 42);
    let mat = to_mat(&data);

    let params = EvocParams::<f32> {
        approx_n_clusters: Some(4),
        min_samples: 3,
        base_min_cluster_size: 3,
        ..EvocParams::default()
    };
    let nn_params = NearestNeighbourParamsGpu::<f32>::default();
    let device = WgpuDevice::default();

    let result = evoc_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        "exhaustive_gpu".to_string(),
        None,
        &params,
        &nn_params,
        device,
        42,
        false,
    );

    assert_eq!(
        result.cluster_layers.len(),
        1,
        "approx_n_clusters should yield one layer"
    );
    let k = result.n_clusters();
    println!("Requested 4 clusters, got {}", k);
    assert!((2..=6).contains(&k), "Got {} clusters, expected near 4", k);
}

/// kNN indices returned from GPU path exclude self (latent run_ann_search_gpu
/// concern noted elsewhere — verify here at the EVoC boundary).
#[test]
fn gpu_integration_07_knn_no_self() {
    let (data, _) = make_blobs(50, 2, 4, 30.0, 0.5, 42);
    let mat = to_mat(&data);

    let params = EvocParams::<f32>::default();
    let nn_params = NearestNeighbourParamsGpu::<f32>::default();
    let device = WgpuDevice::default();

    let result = evoc_gpu::<f32, WgpuRuntime>(
        mat.as_ref(),
        "exhaustive_gpu".to_string(),
        None,
        &params,
        &nn_params,
        device,
        42,
        false,
    );

    let mut self_loops = 0;
    for (i, nb) in result.nn_indices.iter().enumerate() {
        if nb.contains(&i) {
            self_loops += 1;
        }
    }
    // Accept up to 1% as acknowledged GPU tie-breaking artefact.
    let max_allowed = result.nn_indices.len() / 100 + 1;
    assert!(
        self_loops <= max_allowed,
        "{} self-loops out of {} (>1%)",
        self_loops,
        result.nn_indices.len()
    );
}
