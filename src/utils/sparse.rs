//! Sparse structures, conversion and matrix operations for `evoc-rs`.

use crate::prelude::*;
use faer::{Mat, MatRef};
use rayon::prelude::*;

/////////////
// Helpers //
/////////////

/// Sparse accumulator for scatter-gather in CSR matrix multiplication.
///
/// Implements the classic sparse accumulator (SPA) pattern: a dense value and
/// flag array indexed by column, plus a compact list of touched indices.
/// Scatter writes into the dense arrays in O(1); gather collects and resets
/// them in O(nnz log nnz). The flag array makes repeated scatter to the same
/// index an accumulation rather than an overwrite.
struct SpaAcc<T: EvocFloat> {
    /// Dense value buffer; only entries whose flag is set carry meaningful
    /// data.
    values: Vec<T>,
    /// Column indices touched since the last `gather_sorted` call. Used to
    /// iterate and reset without scanning the full buffer.
    indices: Vec<usize>,
    /// Occupancy flags; `flags[i]` is `true` iff `values[i]` has been written
    /// to in the current accumulation round.
    flags: Vec<bool>,
}

impl<T: EvocFloat> SpaAcc<T> {
    /// Create a new sparse accumulator capable of addressing columns `0..size`.
    ///
    /// The internal index buffer is pre-allocated for a tenth of `size` on the
    /// assumption that typical rows are sparse; it will grow if needed.
    ///
    /// ### Params
    ///
    /// * `size` - Number of addressable columns (i.e. the column count of the
    ///   matrix being multiplied into)
    fn new(size: usize) -> Self {
        Self {
            values: vec![T::zero(); size],
            indices: Vec::with_capacity(size / 10),
            flags: vec![false; size],
        }
    }

    /// Accumulate `val` at column `idx`.
    ///
    /// If `idx` has not been touched in the current round it is recorded and
    /// its slot initialised to `val`; otherwise `val` is added to the existing
    /// partial sum.
    ///
    /// ### Params
    ///
    /// * `idx` - Column index to accumulate into
    /// * `val` - Value to add
    ///
    /// # Safety
    ///
    /// `idx` must be less than the `size` passed to `new`. Violating this
    /// causes out-of-bounds writes to `values` and `flags`, which is
    /// undefined behaviour.
    #[inline]
    unsafe fn scatter(&mut self, idx: usize, val: T) {
        unsafe {
            if !*self.flags.get_unchecked(idx) {
                *self.flags.get_unchecked_mut(idx) = true;
                self.indices.push(idx);
                *self.values.get_unchecked_mut(idx) = val;
            } else {
                let cur = *self.values.get_unchecked(idx);
                *self.values.get_unchecked_mut(idx) = cur + val;
            }
        }
    }

    /// Collect all accumulated entries in ascending column order, then reset
    /// the accumulator for reuse.
    ///
    /// Sorting is done in-place on `indices` before reading `values`, so the
    /// output is always ordered by column index. Every touched slot is zeroed
    /// and its flag cleared before returning.
    ///
    /// ### Returns
    ///
    /// A `Vec` of `(column_index, accumulated_value)` pairs sorted by
    /// `column_index`
    #[inline]
    fn gather_sorted(&mut self) -> Vec<(usize, T)> {
        self.indices.sort_unstable();
        let out: Vec<(usize, T)> = self
            .indices
            .iter()
            // Safety: every index in `self.indices` was bounds-checked against
            // `size` at scatter time, so all reads here are in bounds.
            .map(|&i| unsafe { (i, *self.values.get_unchecked(i)) })
            .collect();
        for &i in &self.indices {
            // Safety: same guarantee as above.
            unsafe {
                *self.flags.get_unchecked_mut(i) = false;
                *self.values.get_unchecked_mut(i) = T::zero();
            }
        }
        self.indices.clear();
        out
    }
}

/////////
// COO //
/////////

/// Coordinate list
///
/// Represents the graph in COO (Coordinate) format - tensor-friendly
#[derive(Clone)]
pub struct CoordinateList<T> {
    /// Row index
    pub row_indices: Vec<usize>,
    /// Column index
    pub col_indices: Vec<usize>,
    /// Edge weights
    pub values: Vec<T>,
    /// Number of vertices in the graph
    pub n_samples: usize,
}

/////////
// CSR //
/////////

/// Lightweight CSR for label propagation sparse algebra.
#[derive(Clone, Debug)]
pub struct Csr<T> {
    /// Row index pointers
    pub indptr: Vec<usize>,
    /// Column indices
    pub indices: Vec<usize>,
    /// Data for the sparse matrix
    pub data: Vec<T>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
}

impl<T: EvocFloat> Csr<T> {
    /// Construct a CSR matrix from its raw components.
    ///
    /// No reordering or deduplication is performed; the caller is responsible
    /// for providing a valid CSR representation.
    ///
    /// ### Params
    ///
    /// * `indptr`  - Row pointer array of length `nrows + 1`
    /// * `indices` - Column indices of each stored entry
    /// * `data`    - Values corresponding to each entry in `indices`
    /// * `nrows`   - Number of rows
    /// * `ncols`   - Number of columns
    pub fn new(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<T>,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        debug_assert_eq!(indptr.len(), nrows + 1);
        debug_assert_eq!(indices.len(), data.len());
        debug_assert_eq!(*indptr.last().unwrap(), data.len());
        Self {
            indptr,
            indices,
            data,
            nrows,
            ncols,
        }
    }

    /// Build a square CSR matrix from a COO coordinate list, summing duplicate
    /// entries.
    ///
    /// Triplets are sorted by (row, column) in parallel before assembly.
    /// Consecutive entries sharing the same (row, column) pair are folded into
    /// a single stored value.
    ///
    /// ### Params
    ///
    /// * `coo` - Coordinate list with `n_samples x n_samples` logical shape
    ///
    /// ### Returns
    ///
    /// A square `n_samples x n_samples` CSR matrix
    pub fn from_coo(coo: &CoordinateList<T>) -> Self {
        let n = coo.n_samples;
        let nnz = coo.values.len();
        if nnz == 0 {
            return Self::new(vec![0; n + 1], Vec::new(), Vec::new(), n, n);
        }

        let mut triplets: Vec<(usize, usize, T)> = (0..nnz)
            .map(|i| (coo.row_indices[i], coo.col_indices[i], coo.values[i]))
            .collect();
        triplets.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut data = Vec::with_capacity(nnz);
        let mut indices = Vec::with_capacity(nnz);
        let mut indptr = vec![0usize; n + 1];

        let mut prev_r = usize::MAX;
        let mut prev_c = usize::MAX;
        for &(r, c, v) in &triplets {
            if r == prev_r && c == prev_c {
                let last = data.last().copied().unwrap();
                *data.last_mut().unwrap() = last + v;
            } else {
                data.push(v);
                indices.push(c);
                indptr[r + 1] += 1;
                prev_r = r;
                prev_c = c;
            }
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }

        Self {
            indptr,
            indices,
            data,
            nrows: n,
            ncols: n,
        }
    }

    /// Build a partition indicator matrix of shape `n_points x n_parts`.
    ///
    /// Row `i` contains a single `1.0` at column `partition[i]`, encoding a
    /// hard cluster assignment as a sparse one-hot matrix.
    ///
    /// ### Params
    ///
    /// * `partition` - Slice of length `n_points` where `partition[i]` is the
    ///   part index assigned to point `i`
    /// * `n_parts`   - Total number of parts (column count)
    pub fn from_partition(partition: &[usize], n_parts: usize) -> Self {
        let n = partition.len();
        Self {
            indptr: (0..=n).collect(),
            indices: partition.to_vec(),
            data: vec![T::one(); n],
            nrows: n,
            ncols: n_parts,
        }
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Transpose into a new CSR matrix.
    ///
    /// Constructs the transpose via a two-pass counting sort: the first pass
    /// builds column counts to allocate `indptr`; the second scatters each
    /// entry to its transposed position using per-column cursors.
    ///
    /// ### Returns
    ///
    /// A new `ncols x nrows` CSR matrix equal to `self^T`
    pub fn transpose(&self) -> Self {
        let nnz = self.nnz();
        let mut col_count = vec![0usize; self.ncols];
        for &c in &self.indices {
            col_count[c] += 1;
        }

        let mut indptr = vec![0usize; self.ncols + 1];
        for i in 0..self.ncols {
            indptr[i + 1] = indptr[i] + col_count[i];
        }

        let mut data = vec![T::zero(); nnz];
        let mut indices = vec![0usize; nnz];
        let mut cursor = indptr[..self.ncols].to_vec();

        for row in 0..self.nrows {
            for idx in self.indptr[row]..self.indptr[row + 1] {
                let col = self.indices[idx];
                let pos = cursor[col];
                data[pos] = self.data[idx];
                indices[pos] = row;
                cursor[col] += 1;
            }
        }

        Self {
            indptr,
            indices,
            data,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }

    /// Sparse-sparse matrix multiplication: `self` (m x k) * `other` (k x n)
    /// -> (m x n).
    ///
    /// ### Params
    ///
    /// * `other` - Right-hand operand; its row count must equal `self.ncols`
    ///
    /// ### Returns
    ///
    /// A new `m x n` CSR matrix
    pub fn matmul(&self, other: &Csr<T>) -> Self {
        assert_eq!(
            self.ncols, other.nrows,
            "Dimension mismatch: ({} x {}) * ({} x {})",
            self.nrows, self.ncols, other.nrows, other.ncols
        );

        let m = self.nrows;
        let n = other.ncols;

        let row_results: Vec<Vec<(usize, T)>> = (0..m)
            .into_par_iter()
            .map(|i| {
                let mut acc = SpaAcc::new(n);
                for a_idx in self.indptr[i]..self.indptr[i + 1] {
                    let k = self.indices[a_idx];
                    let a_val = self.data[a_idx];
                    for b_idx in other.indptr[k]..other.indptr[k + 1] {
                        unsafe {
                            acc.scatter(other.indices[b_idx], a_val * other.data[b_idx]);
                        }
                    }
                }
                acc.gather_sorted()
            })
            .collect();

        let total_nnz: usize = row_results.iter().map(|r| r.len()).sum();
        let mut data = Vec::with_capacity(total_nnz);
        let mut indices = Vec::with_capacity(total_nnz);
        let mut indptr = Vec::with_capacity(m + 1);
        indptr.push(0);

        for row in row_results {
            for (col, val) in row {
                indices.push(col);
                data.push(val);
            }
            indptr.push(data.len());
        }

        Self {
            indptr,
            indices,
            data,
            nrows: m,
            ncols: n,
        }
    }

    /// Element-wise (Hadamard) product of two matrices with identical shape.
    ///
    /// ### Params
    ///
    /// * `other` - Right-hand operand; must have the same shape as `self` and
    ///   sorted column indices per row
    ///
    /// ### Returns
    ///
    /// A new CSR matrix containing only the entries where both operands are
    /// non-zero
    pub fn elementwise_mul(&self, other: &Csr<T>) -> Self {
        assert_eq!(
            (self.nrows, self.ncols),
            (other.nrows, other.ncols),
            "Shape mismatch for element-wise multiply"
        );

        let mut indptr = vec![0usize; self.nrows + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for i in 0..self.nrows {
            let (mut p, end_p) = (self.indptr[i], self.indptr[i + 1]);
            let (mut q, end_q) = (other.indptr[i], other.indptr[i + 1]);
            while p < end_p && q < end_q {
                let ci = self.indices[p];
                let cj = other.indices[q];
                match ci.cmp(&cj) {
                    std::cmp::Ordering::Equal => {
                        indices.push(ci);
                        data.push(self.data[p] * other.data[q]);
                        p += 1;
                        q += 1;
                    }
                    std::cmp::Ordering::Less => p += 1,
                    std::cmp::Ordering::Greater => q += 1,
                }
            }
            indptr[i + 1] = data.len();
        }

        Self {
            indptr,
            indices,
            data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Column-wise L2 normalisation.
    ///
    /// Each column is scaled by the reciprocal of its L2 norm. Columns with
    /// zero norm are left unchanged (scale factor of 1).
    ///
    /// ### Returns
    ///
    /// A new CSR matrix with the same sparsity pattern and unit-norm columns
    pub fn normalise_cols_l2(&self) -> Self {
        let mut col_sq = vec![T::zero(); self.ncols];
        for (idx, &v) in self.data.iter().enumerate() {
            let c = self.indices[idx];
            col_sq[c] += v * v;
        }

        let col_inv: Vec<T> = col_sq
            .iter()
            .map(|&sq| {
                let norm = sq.sqrt();
                if norm > T::zero() {
                    T::one() / norm
                } else {
                    T::one()
                }
            })
            .collect();

        let new_data: Vec<T> = self
            .data
            .iter()
            .enumerate()
            .map(|(idx, &v)| v * col_inv[self.indices[idx]])
            .collect();

        Self {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: new_data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Row-wise L1 normalisation.
    ///
    /// Each row is scaled by the reciprocal of the sum of absolute values of
    /// its entries. Rows with zero norm are left unchanged.
    ///
    /// ### Returns
    ///
    /// A new CSR matrix with the same sparsity pattern and unit-L1-norm rows
    pub fn normalise_rows_l1(&self) -> Self {
        let mut new_data = self.data.clone();
        for i in 0..self.nrows {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            let mut norm = T::zero();
            for idx in start..end {
                norm += self.data[idx].abs();
            }
            if norm > T::zero() {
                let inv = T::one() / norm;
                for idx in start..end {
                    new_data[idx] = new_data[idx] * inv;
                }
            }
        }

        Self {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: new_data,
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }

    /// Clamp all stored values to the closed interval `[lo, hi]`.
    ///
    /// ### Params
    ///
    /// * `lo` - Lower bound
    /// * `hi` - Upper bound
    pub fn clip_values(&mut self, lo: T, hi: T) {
        for d in &mut self.data {
            if *d < lo {
                *d = lo;
            } else if *d > hi {
                *d = hi;
            }
        }
    }

    /// Convert to an adjacency list representation.
    ///
    /// Each entry `graph[i]` is a `Vec` of `(column, value)` pairs for row
    /// `i`, suitable for consumption by `evoc_embedding`.
    ///
    /// ### Returns
    ///
    /// A `Vec` of length `nrows`, where `graph[i]` contains the neighbours and
    /// edge weights of node `i`
    pub fn to_adjacency_list(&self) -> Vec<Vec<(usize, T)>> {
        (0..self.nrows)
            .map(|i| {
                (self.indptr[i]..self.indptr[i + 1])
                    .map(|idx| (self.indices[idx], self.data[idx]))
                    .collect()
            })
            .collect()
    }

    /// Sparse-dense matrix multiplication: `self` (m x k) * `rhs` (k x d)
    /// -> (m x d).
    ///
    /// ### Params
    ///
    /// * `rhs` - Dense right-hand operand; its row count must equal
    ///   `self.ncols`
    ///
    /// ### Returns
    ///
    /// A dense `m x d` matrix
    pub fn matmul_dense(&self, rhs: &MatRef<T>) -> Mat<T> {
        assert_eq!(
            self.ncols,
            rhs.nrows(),
            "Dimension mismatch: CSR cols {} vs Mat rows {}",
            self.ncols,
            rhs.nrows()
        );

        let d = rhs.ncols();
        let rows: Vec<Vec<T>> = (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![T::zero(); d];
                for idx in self.indptr[i]..self.indptr[i + 1] {
                    let j = self.indices[idx];
                    let v = self.data[idx];
                    for k in 0..d {
                        row[k] += v * rhs[(j, k)];
                    }
                }
                row
            })
            .collect();

        Mat::from_fn(self.nrows, d, |i, j| rows[i][j])
    }

    /// Convert to a dense `faer::Mat`, filling structural zeros explicitly.
    ///
    /// ### Returns
    ///
    /// A dense `nrows x ncols` matrix
    pub fn to_dense(&self) -> Mat<T> {
        let mut dense = Mat::zeros(self.nrows, self.ncols);
        for i in 0..self.nrows {
            for idx in self.indptr[i]..self.indptr[i + 1] {
                dense[(i, self.indices[idx])] = self.data[idx];
            }
        }
        dense
    }
}

///////////////////////////
// Conversion utilities  //
///////////////////////////

/// Pack a row-major `Vec<Vec<T>>` into a `faer::Mat`.
///
/// All inner `Vec`s must have the same length. An empty outer slice produces a
/// `0 x 0` matrix.
///
/// ### Params
///
/// * `rows` - Slice of rows, each of length `d`
///
/// ### Returns
///
/// A `rows.len() x d` matrix
pub fn vecs_to_mat<T: EvocFloat>(rows: &[Vec<T>]) -> Mat<T> {
    let n = rows.len();
    if n == 0 {
        return Mat::zeros(0, 0);
    }
    let d = rows[0].len();
    Mat::from_fn(n, d, |i, j| rows[i][j])
}

/// Unpack a `faer::Mat` into a row-major `Vec<Vec<T>>`.
///
/// ### Params
///
/// * `mat` - Matrix to unpack
///
/// ### Returns
///
/// A `Vec` of length `nrows`, each inner `Vec` of length `ncols`
pub fn mat_to_vecs<T: EvocFloat>(mat: &Mat<T>) -> Vec<Vec<T>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| mat[(i, j)]).collect())
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a small 3x3 CSR.
    ///
    /// ```text
    /// [[1, 0, 2],
    ///  [0, 3, 0],
    ///  [4, 0, 5]]
    /// ```
    fn make_3x3() -> Csr<f64> {
        Csr::new(
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            3,
            3,
        )
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn from_coo_basic() {
        let coo = CoordinateList {
            row_indices: vec![0, 0, 1, 2, 2],
            col_indices: vec![0, 2, 1, 0, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            n_samples: 3,
        };
        let csr = Csr::from_coo(&coo);
        assert_eq!(csr.nrows, 3);
        assert_eq!(csr.ncols, 3);
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.indptr, vec![0, 2, 3, 5]);
        assert_eq!(csr.indices, vec![0, 2, 1, 0, 2]);
        assert_eq!(csr.data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn from_coo_duplicates_summed() {
        let coo = CoordinateList {
            row_indices: vec![0, 0, 0],
            col_indices: vec![1, 1, 2],
            values: vec![1.0, 3.0, 5.0],
            n_samples: 2,
        };
        let csr = Csr::from_coo(&coo);
        // (0,1) should be 1+3=4, (0,2) should be 5
        assert_eq!(csr.indptr, vec![0, 2, 2]);
        assert_eq!(csr.indices, vec![1, 2]);
        assert!(approx_eq(csr.data[0], 4.0));
        assert!(approx_eq(csr.data[1], 5.0));
    }

    #[test]
    fn from_coo_empty() {
        let coo: CoordinateList<f64> = CoordinateList {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            n_samples: 5,
        };
        let csr = Csr::from_coo(&coo);
        assert_eq!(csr.nrows, 5);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.indptr, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn from_partition_basic() {
        let part = vec![2, 0, 1, 2];
        let csr = Csr::<f64>::from_partition(&part, 3);
        assert_eq!(csr.nrows, 4);
        assert_eq!(csr.ncols, 3);
        assert_eq!(csr.nnz(), 4);
        // Row 0 -> col 2, row 1 -> col 0, etc.
        assert_eq!(csr.indices, vec![2, 0, 1, 2]);
        assert!(csr.data.iter().all(|&v| approx_eq(v, 1.0)));
    }

    #[test]
    fn transpose_roundtrip() {
        let a = make_3x3();
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 3);
        assert_eq!(at.nnz(), 5);

        // A^T[0] should be cols [0, 2] with vals [1, 4]
        let row0: Vec<(usize, f64)> = (at.indptr[0]..at.indptr[1])
            .map(|idx| (at.indices[idx], at.data[idx]))
            .collect();
        assert_eq!(row0, vec![(0, 1.0), (2, 4.0)]);

        // Double transpose should recover original
        let att = at.transpose();
        assert_eq!(att.indptr, a.indptr);
        assert_eq!(att.indices, a.indices);
        assert_eq!(att.data, a.data);
    }

    #[test]
    fn transpose_non_square() {
        // 2x3: [[1, 2, 0], [0, 0, 3]]
        let m = Csr::new(vec![0, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0], 2, 3);
        let mt = m.transpose();
        assert_eq!(mt.nrows, 3);
        assert_eq!(mt.ncols, 2);
        // T[0] = [1, 0], T[1] = [2, 0], T[2] = [0, 3]
        assert_eq!(mt.indptr, vec![0, 1, 2, 3]);
        assert_eq!(mt.indices, vec![0, 0, 1]);
        assert_eq!(mt.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_identity() {
        let a = make_3x3();
        // 3x3 identity
        let eye = Csr::new(vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0], 3, 3);
        let result = a.matmul(&eye);
        assert_eq!(result.data, a.data);
        assert_eq!(result.indices, a.indices);
    }

    #[test]
    fn matmul_a_times_at() {
        let a = make_3x3();
        let at = a.transpose();
        let aat = a.matmul(&at);
        let dense = aat.to_dense();

        // A * A^T = [[5, 0, 14], [0, 9, 0], [14, 0, 41]]
        assert!(approx_eq(dense[(0, 0)], 5.0));
        assert!(approx_eq(dense[(0, 1)], 0.0));
        assert!(approx_eq(dense[(0, 2)], 14.0));
        assert!(approx_eq(dense[(1, 1)], 9.0));
        assert!(approx_eq(dense[(2, 0)], 14.0));
        assert!(approx_eq(dense[(2, 2)], 41.0));
    }

    #[test]
    fn matmul_non_square() {
        // (2x3) * (3x2)
        let a = Csr::new(vec![0, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0], 2, 3);
        let b = Csr::new(vec![0, 1, 2, 3], vec![0, 1, 0], vec![4.0, 5.0, 6.0], 3, 2);
        let c = a.matmul(&b);
        assert_eq!(c.nrows, 2);
        assert_eq!(c.ncols, 2);
        let dense = c.to_dense();
        // Row 0: 1*[4,0] + 2*[0,5] = [4, 10]
        // Row 1: 3*[6,0] = [18, 0]
        assert!(approx_eq(dense[(0, 0)], 4.0));
        assert!(approx_eq(dense[(0, 1)], 10.0));
        assert!(approx_eq(dense[(1, 0)], 18.0));
        assert!(approx_eq(dense[(1, 1)], 0.0));
    }

    #[test]
    fn matmul_dense_basic() {
        let a = make_3x3();
        // Dense 3x2: [[1, 0], [0, 1], [1, 1]]
        let rhs = Mat::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) | (2, 0) | (2, 1) => 1.0_f64,
            _ => 0.0,
        });
        let result = a.matmul_dense(&rhs.as_ref());
        // Row 0: 1*[1,0] + 2*[1,1] = [3, 2]
        // Row 1: 3*[0,1] = [0, 3]
        // Row 2: 4*[1,0] + 5*[1,1] = [9, 5]
        assert!(approx_eq(result[(0, 0)], 3.0));
        assert!(approx_eq(result[(0, 1)], 2.0));
        assert!(approx_eq(result[(1, 0)], 0.0));
        assert!(approx_eq(result[(1, 1)], 3.0));
        assert!(approx_eq(result[(2, 0)], 9.0));
        assert!(approx_eq(result[(2, 1)], 5.0));
    }

    #[test]
    fn elementwise_mul_with_transpose() {
        let a = make_3x3();
        let at = a.transpose();
        let h = a.elementwise_mul(&at);
        let dense = h.to_dense();

        // A .* A^T:
        // (0,0): 1*1=1, (0,2): 2*4=8
        // (1,1): 3*3=9
        // (2,0): 4*2=8, (2,2): 5*5=25
        assert!(approx_eq(dense[(0, 0)], 1.0));
        assert!(approx_eq(dense[(0, 2)], 8.0));
        assert!(approx_eq(dense[(1, 1)], 9.0));
        assert!(approx_eq(dense[(2, 0)], 8.0));
        assert!(approx_eq(dense[(2, 2)], 25.0));
        assert_eq!(h.nnz(), 5);
    }

    #[test]
    fn elementwise_mul_disjoint() {
        // No overlapping entries -> empty result
        let a = Csr::new(vec![0, 1, 1], vec![0], vec![5.0], 2, 2);
        let b = Csr::new(vec![0, 0, 1], vec![1], vec![7.0], 2, 2);
        let h = a.elementwise_mul(&b);
        assert_eq!(h.nnz(), 0);
    }

    #[test]
    fn normalise_cols_l2_unit_norms() {
        // [[1, 0], [0, 2], [3, 4]]
        let m = Csr::new(
            vec![0, 1, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            3,
            2,
        );
        let normed = m.normalise_cols_l2();

        // Check column norms are 1
        let mut col_sq = [0.0f64; 2];
        for (idx, &v) in normed.data.iter().enumerate() {
            col_sq[normed.indices[idx]] += v * v;
        }
        assert!(approx_eq(col_sq[0].sqrt(), 1.0));
        assert!(approx_eq(col_sq[1].sqrt(), 1.0));

        // Check specific values
        let c0_norm = (1.0f64 + 9.0).sqrt(); // sqrt(10)
        let c1_norm = (4.0f64 + 16.0).sqrt(); // sqrt(20)
        assert!(approx_eq(normed.data[0], 1.0 / c0_norm));
        assert!(approx_eq(normed.data[1], 2.0 / c1_norm));
        assert!(approx_eq(normed.data[2], 3.0 / c0_norm));
        assert!(approx_eq(normed.data[3], 4.0 / c1_norm));
    }

    #[test]
    fn normalise_cols_l2_empty_column() {
        // Column 1 has no entries -> should not panic
        let m = Csr::new(vec![0, 1, 1], vec![0], vec![3.0], 2, 2);
        let normed = m.normalise_cols_l2();
        assert!(approx_eq(normed.data[0], 1.0)); // 3/3 = 1
    }

    #[test]
    fn normalise_rows_l1_unit_sums() {
        let m = Csr::new(
            vec![0, 1, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
            3,
            2,
        );
        let normed = m.normalise_rows_l1();

        // Row sums should be 1
        for i in 0..normed.nrows {
            let sum: f64 = normed.data[normed.indptr[i]..normed.indptr[i + 1]]
                .iter()
                .map(|v: &f64| v.abs())
                .sum();
            assert!(approx_eq(sum, 1.0));
        }

        // Row 0: [1] -> [1]
        assert!(approx_eq(normed.data[0], 1.0));
        // Row 1: [2] -> [1]
        assert!(approx_eq(normed.data[1], 1.0));
        // Row 2: [3, 4] -> [3/7, 4/7]
        assert!(approx_eq(normed.data[2], 3.0 / 7.0));
        assert!(approx_eq(normed.data[3], 4.0 / 7.0));
    }

    #[test]
    fn normalise_rows_l1_empty_row() {
        let m = Csr::new(vec![0, 0, 1], vec![0], vec![5.0], 2, 2);
        let normed = m.normalise_rows_l1();
        // Row 0 is empty -> should not panic
        assert!(approx_eq(normed.data[0], 1.0));
    }

    #[test]
    fn clip_values_basic() {
        let mut m = Csr::new(vec![0, 3], vec![0, 1, 2], vec![-1.0, 0.5, 2.0], 1, 3);
        m.clip_values(0.0, 1.0);
        assert!(approx_eq(m.data[0], 0.0));
        assert!(approx_eq(m.data[1], 0.5));
        assert!(approx_eq(m.data[2], 1.0));
    }

    #[test]
    fn to_adjacency_list_roundtrip() {
        let a = make_3x3();
        let adj = a.to_adjacency_list();
        assert_eq!(adj.len(), 3);
        assert_eq!(adj[0], vec![(0, 1.0), (2, 2.0)]);
        assert_eq!(adj[1], vec![(1, 3.0)]);
        assert_eq!(adj[2], vec![(0, 4.0), (2, 5.0)]);
    }

    #[test]
    fn to_dense_roundtrip() {
        let a = make_3x3();
        let d = a.to_dense();
        assert!(approx_eq(d[(0, 0)], 1.0));
        assert!(approx_eq(d[(0, 1)], 0.0));
        assert!(approx_eq(d[(0, 2)], 2.0));
        assert!(approx_eq(d[(1, 1)], 3.0));
        assert!(approx_eq(d[(2, 0)], 4.0));
        assert!(approx_eq(d[(2, 2)], 5.0));
    }

    #[test]
    fn vecs_mat_roundtrip() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let mat = vecs_to_mat(&rows);
        let back = mat_to_vecs(&mat);
        assert_eq!(rows, back);
    }
}
