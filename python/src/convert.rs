//! numpy in, numpy out.
//!
//! Input goes through `PyReadonlyArray2::as_slice`, which feeds the crate's
//! `impl EvocMatrix<T> for (&[T], usize, usize)` with no copy. Output is
//! rectangular by construction: every cluster layer and every strength vector
//! carries one entry per sample, so the layers pack into `(n_layers, n)`
//! without padding. The kNN rows are the exception, `run_ann_search` can hand
//! back a short row for an approximate backend, so those are densified.

use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/////////////////
// Fill values //
/////////////////

/// Neighbour slot that was never populated.
///
/// `-1` is the pynndescent and umap convention. It survives fancy indexing as
/// "the last row" rather than raising, so callers must mask on `>= 0` before
/// slicing with it.
const NO_NEIGHBOUR: i64 = -1;

/// The kNN graph on its way back to Python: `(indices, distances)`.
pub(crate) type KnnArrays<'py, T> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<T>>);

////////////
// Inputs //
////////////

/// Borrow a C-contiguous 2-D array as `(data, n_rows, n_cols)`.
///
/// The returned slice borrows from `a`, so `a` must outlive any
/// `Python::detach` the slice is passed into. Keep this call outside the
/// closure.
///
/// ### Params
///
/// * `a` - Read-only view of a 2-D numpy array.
///
/// ### Returns
///
/// `(data, n_rows, n_cols)`, row-major, in the shape the crate's `EvocMatrix`
/// tuple impl wants. Errors if the array is not C-contiguous; the Python layer
/// runs `np.ascontiguousarray` first, so this is a backstop.
pub(crate) fn flat<'a, T: Element>(
    a: &'a PyReadonlyArray2<'_, T>,
) -> PyResult<(&'a [T], usize, usize)> {
    let shape = a.shape();
    let (n, dim) = (shape[0], shape[1]);
    let data = a.as_slice().map_err(|_| {
        PyValueError::new_err("array must be C-contiguous; use np.ascontiguousarray")
    })?;
    Ok((data, n, dim))
}

/////////////
// Outputs //
/////////////

/// Stack equal-length rows into a dense `(n_rows, row_len)` array.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `rows` - The rows to stack. Every row must have the same length.
/// * `convert` - Applied to each entry on the way in.
///
/// ### Returns
///
/// A `(rows.len(), row_len)` array, or a `(0, 0)` one when `rows` is empty.
/// `into_pyarray` moves the buffer into the numpy object and `reshape` returns
/// a view, so neither copies.
fn stack<'py, S, D>(
    py: Python<'py>,
    rows: &[Vec<S>],
    convert: impl Fn(S) -> D,
) -> PyResult<Bound<'py, PyArray2<D>>>
where
    S: Copy,
    D: Element,
{
    let row_len = rows.first().map_or(0, |r| r.len());
    let out: Vec<D> = rows
        .iter()
        .flat_map(|row| row.iter().copied().map(&convert))
        .collect();
    out.into_pyarray(py).reshape([rows.len(), row_len])
}

/// Pack the per-layer cluster labels into a dense `(n_layers, n)` `int64`
/// array.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `layers` - One label vector per layer, finest first. `-1` is noise.
///
/// ### Returns
///
/// A `(n_layers, n)` array of labels.
pub(crate) fn pack_layers<'py>(
    py: Python<'py>,
    layers: &[Vec<i64>],
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    stack(py, layers, |v| v)
}

/// Pack the per-layer membership strengths into a dense `(n_layers, n)` array.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `strengths` - One strength vector per layer, aligned with the layers.
///
/// ### Returns
///
/// A `(n_layers, n)` array of `T`, matching the element type the clustering
/// ran at rather than widening to `f64`.
pub(crate) fn pack_strengths<'py, T>(
    py: Python<'py>,
    strengths: &[Vec<T>],
) -> PyResult<Bound<'py, PyArray2<T>>>
where
    T: Element + Copy,
{
    stack(py, strengths, |v| v)
}

/// Move the per-layer persistence scores into a `(n_layers,)` `float64` array.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `scores` - One score per layer.
///
/// ### Returns
///
/// A `(n_layers,)` array. The scores are `f64` in the core regardless of the
/// clustering's element type, so no conversion happens here.
pub(crate) fn pack_scores<'py>(py: Python<'py>, scores: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    scores.into_pyarray(py)
}

/// Flatten ragged rows into a dense `n * k` buffer, padding short rows.
///
/// One allocation, no reallocation: the buffer is filled with `fill` up front
/// and each row's prefix is written over it. Rows longer than `k` are
/// truncated, which cannot happen today but keeps a future over-returning
/// backend from panicking here.
fn densify<S, D>(rows: &[Vec<S>], k: usize, fill: D, convert: impl Fn(S) -> D) -> Vec<D>
where
    S: Copy,
    D: Copy,
{
    let mut out = vec![fill; rows.len() * k];
    for (i, row) in rows.iter().enumerate() {
        let slots = &mut out[i * k..(i + 1) * k];
        for (slot, &value) in slots.iter_mut().zip(row.iter().take(k)) {
            *slot = convert(value);
        }
    }
    out
}

/// Pack the kNN graph into dense `(n, k)` index and distance arrays.
///
/// Distance padding is `+inf` rather than NaN: it keeps each row totally
/// ordered, so `argsort`, `min` and the sortedness callers rely on all still
/// hold. NaN would poison all three.
///
/// ### Params
///
/// * `py` - Attached interpreter token.
/// * `indices` - Neighbour indices, one row per sample, self excluded.
/// * `distances` - Neighbour distances, aligned with `indices`.
///
/// ### Returns
///
/// `(indices, distances)` as `(n, k)` arrays, `k` taken from the longest row.
/// Short rows are padded with [`NO_NEIGHBOUR`] and `+inf`.
pub(crate) fn pack_knn<'py, T>(
    py: Python<'py>,
    indices: &[Vec<usize>],
    distances: &[Vec<T>],
) -> PyResult<KnnArrays<'py, T>>
where
    T: Element + num_traits::Float,
{
    let n = indices.len();
    let k = indices.iter().map(Vec::len).max().unwrap_or(0);

    let idx = densify(indices, k, NO_NEIGHBOUR, |v| v as i64);
    let dist = densify(distances, k, T::infinity(), |v| v);

    Ok((
        idx.into_pyarray(py).reshape([n, k])?,
        dist.into_pyarray(py).reshape([n, k])?,
    ))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_densify_pads_short_rows() {
        let rows = vec![vec![1usize, 2], vec![3]];
        let out = densify(&rows, 3, NO_NEIGHBOUR, |v| v as i64);
        assert_eq!(out, vec![1, 2, -1, 3, -1, -1]);
    }

    #[test]
    fn test_densify_truncates_long_rows() {
        let rows = vec![vec![1usize, 2, 3, 4]];
        let out = densify(&rows, 2, NO_NEIGHBOUR, |v| v as i64);
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn test_densify_empty_row_is_all_padding() {
        let rows: Vec<Vec<usize>> = vec![vec![]];
        let out = densify(&rows, 3, NO_NEIGHBOUR, |v| v as i64);
        assert_eq!(out, vec![-1, -1, -1]);
    }

    #[test]
    fn test_densify_distance_padding_stays_ordered() {
        let rows = vec![vec![0.5f32, 1.5]];
        let out = densify(&rows, 4, f32::INFINITY, |v| v);
        assert_eq!(out[..2], [0.5, 1.5]);
        assert!(out.windows(2).all(|w| w[0] <= w[1]));
    }
}
