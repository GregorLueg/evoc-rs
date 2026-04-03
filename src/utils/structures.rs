//! Shared structures across the crate like COOrdinate formats, compressed
//! sparse data formats, etc.

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
