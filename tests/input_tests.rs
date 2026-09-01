//! The representation a caller happens to hold must not change the answer.
//! Every test builds the same data in several shapes and asserts they agree.

mod commons;
use commons::*;

use evoc_rs::prelude::*;
use evoc_rs::{EvocParams, evoc};
use faer::{Mat, MatRef};

/////////////
// Helpers //
/////////////

/// Three well-separated blobs as a row-major buffer plus its shape.
///
/// ### Returns
///
/// `(flat, n_samples, n_features)`, row-major.
fn fixture() -> (Vec<f64>, usize, usize) {
    let (data, _) = make_blobs(40, 3, 8, 30.0, 0.5, 42);
    let n = data.len();
    let dim = data[0].len();
    let flat: Vec<f64> = data.into_iter().flatten().collect();
    (flat, n, dim)
}

/// Run `evoc` with the deterministic settings these tests need.
///
/// `"exhaustive"` removes ANN randomness, so any disagreement between two
/// representations is the conversion and nothing else.
fn cluster(data: impl EvocMatrix<f64>) -> Vec<i64> {
    let params = EvocParams {
        n_neighbours: 10,
        n_epochs: 20,
        ..Default::default()
    };
    let result = evoc(
        data,
        "exhaustive".to_string(),
        None,
        &params,
        &NearestNeighbourParamsEvoc::default(),
        42,
        0,
    )
    .expect("evoc should succeed on the fixture");
    result.best_labels().to_vec()
}

/////////////////
// Conversions //
/////////////////

#[test]
fn input_faer_and_flat_triple_agree() {
    let (flat, n, dim) = fixture();
    let mat = MatRef::from_row_major_slice(&flat, n, dim);

    let triple = (flat.as_slice(), n, dim);
    let from_faer = mat.to_mat_input();
    let from_flat = triple.to_mat_input();

    let a = from_faer.as_mat_ref();
    let b = from_flat.as_mat_ref();

    assert_eq!(a.nrows(), b.nrows());
    assert_eq!(a.ncols(), b.ncols());
    for i in 0..n {
        for j in 0..dim {
            assert_eq!(a[(i, j)], b[(i, j)]);
        }
    }
}

#[test]
fn input_owned_flat_triple_matches_borrowed() {
    let (flat, n, dim) = fixture();

    let borrowed_src = (flat.as_slice(), n, dim);
    let owned_src = (flat.clone(), n, dim);
    let borrowed = borrowed_src.to_mat_input();
    let owned = owned_src.to_mat_input();

    for i in 0..n {
        for j in 0..dim {
            assert_eq!(borrowed.as_mat_ref()[(i, j)], owned.as_mat_ref()[(i, j)]);
        }
    }
}

/// faer is column-major internally, so an owned `Mat` built row-by-row must
/// still come back in the orientation the flat triple describes.
#[test]
fn input_owned_mat_matches_flat_triple() {
    let (flat, n, dim) = fixture();
    let mat = Mat::from_fn(n, dim, |i, j| flat[i * dim + j]);

    let triple = (flat.as_slice(), n, dim);
    let from_mat = mat.to_mat_input();
    let from_flat = triple.to_mat_input();

    for i in 0..n {
        for j in 0..dim {
            assert_eq!(from_mat.as_mat_ref()[(i, j)], from_flat.as_mat_ref()[(i, j)]);
        }
    }
}

#[test]
#[should_panic(expected = "does not match shape")]
fn input_flat_triple_rejects_wrong_shape() {
    let (flat, n, dim) = fixture();
    let bad = (flat.as_slice(), n + 1, dim);
    let _ = bad.to_mat_input();
}

//////////////////
// End to end   //
//////////////////

#[test]
fn input_evoc_agrees_across_representations() {
    let (flat, n, dim) = fixture();
    let mat = Mat::from_fn(n, dim, |i, j| flat[i * dim + j]);

    let from_faer = cluster(mat.as_ref());
    let from_flat = cluster((flat.as_slice(), n, dim));
    let from_owned = cluster((flat.clone(), n, dim));
    let from_mat = cluster(&mat);

    assert_eq!(from_faer, from_flat, "faer and flat slice disagree");
    assert_eq!(from_faer, from_owned, "faer and owned triple disagree");
    assert_eq!(from_faer, from_mat, "faer ref and Mat disagree");
}

/////////////
// ndarray //
/////////////

#[cfg(feature = "ndarray")]
mod ndarray_inputs {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn input_ndarray_matches_faer() {
        let (flat, n, dim) = fixture();
        let arr = Array2::from_shape_vec((n, dim), flat.clone()).unwrap();
        let mat = MatRef::from_row_major_slice(&flat, n, dim);

        let from_arr = arr.to_mat_input();
        let from_faer = mat.to_mat_input();

        for i in 0..n {
            for j in 0..dim {
                assert_eq!(
                    from_arr.as_mat_ref()[(i, j)],
                    from_faer.as_mat_ref()[(i, j)]
                );
            }
        }
    }

    /// A transposed view has no row-major stride, so it must take the `Owned`
    /// arm rather than being silently reinterpreted.
    #[test]
    fn input_transposed_ndarray_view_is_not_reinterpreted() {
        let (flat, n, dim) = fixture();
        // Build the transpose as a real array, then view it back transposed:
        // the result is logically (n, dim) but strided.
        let column_major = Array2::from_shape_vec((dim, n), {
            let mut out = vec![0.0f64; n * dim];
            for i in 0..n {
                for j in 0..dim {
                    out[j * n + i] = flat[i * dim + j];
                }
            }
            out
        })
        .unwrap();
        let view = column_major.t();

        assert!(
            view.to_slice().is_none(),
            "the view must be non-contiguous for this test to mean anything"
        );

        let input = view.to_mat_input();
        let mat = input.as_mat_ref();
        for i in 0..n {
            for j in 0..dim {
                assert_eq!(mat[(i, j)], flat[i * dim + j]);
            }
        }
    }

    #[test]
    fn input_evoc_agrees_for_ndarray() {
        let (flat, n, dim) = fixture();
        let arr = Array2::from_shape_vec((n, dim), flat.clone()).unwrap();

        let from_flat = cluster((flat.as_slice(), n, dim));
        let from_arr = cluster(&arr);

        assert_eq!(from_flat, from_arr, "flat slice and ndarray disagree");
    }
}
