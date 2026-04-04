//! Disjoint set helpers

///////////////////////////
// DisjointSet structure //
///////////////////////////

/// Union-Find with union by rank and path halving.
///
/// Standard disjoint set forest supporting near-constant-time `find` and
/// `union` operations (amortised inverse Ackermann). Used internally for
/// connected component tracking during clustering.
pub struct DisjointSet {
    /// Parent pointers; `parent[x] == x` indicates a root.
    parent: Vec<usize>,
    /// Upper bound on subtree depth, used to keep merges balanced.
    rank: Vec<usize>,
}

impl DisjointSet {
    /// Create a new disjoint set with `n` elements, each in its own singleton
    /// set.
    ///
    /// ### Params
    ///
    /// * `n` - Number of elements (indexed `0..n`)
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Find the representative (root) of the set containing `x`.
    ///
    /// Applies path halving: each traversed node is repointed to its
    /// grandparent, flattening the tree over successive calls.
    ///
    /// ### Params
    ///
    /// * `x` - Element to look up
    ///
    /// ### Returns
    ///
    /// Root index of the component containing `x`
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merge the sets containing `x` and `y` by rank.
    ///
    /// The shorter tree is attached under the taller one to keep depth
    /// bounded.
    ///
    /// ### Params
    ///
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// ### Returns
    ///
    /// `true` if `x` and `y` were in different sets (i.e. a merge actually
    /// happened), `false` if they were already connected.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }

        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
        true
    }

    /// Check whether `x` and `y` belong to the same set.
    ///
    /// ### Params
    ///
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// ### Returns
    ///
    /// `true` if both elements share the same root
    pub fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }
}

//////////////////////
// SizedDisjointSet //
//////////////////////

/// Union-Find that tracks component sizes.
///
/// Same structure as [`DisjointSet`] but uses union by size instead of rank,
/// and exposes component sizes. Used during linkage tree construction where
/// merge sizes feed into distance/weight calculations.
///
/// ### Fields
///
/// * `parent` - Parent pointers; `parent[x] == x` indicates a root
/// * `size` - Number of elements in the subtree rooted at each node; only
///   meaningful at root nodes
pub struct SizedDisjointSet {
    /// Parent pointers; `parent[x] == x` indicates a root.
    parent: Vec<usize>,
    /// Component size; only valid at root nodes.
    size: Vec<usize>,
}

impl SizedDisjointSet {
    /// Create with `n` singleton elements, each of size 1.
    ///
    /// ### Params
    ///
    /// * `n` - Number of elements (indexed `0..n`)
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    /// Find the representative of the set containing `x`, with path halving.
    ///
    /// ### Params
    ///
    /// * `x` - Element to look up
    ///
    /// ### Returns
    ///
    /// Root index of the component containing `x`
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merge the sets containing `x` and `y` by size.
    ///
    /// The smaller component is attached under the larger one.
    ///
    /// ### Params
    ///
    /// * `x` - First element
    /// * `y` - Second element
    ///
    /// ### Returns
    ///
    /// `true` if a merge occurred, `false` if already in the same set.
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }

        if self.size[rx] < self.size[ry] {
            self.parent[rx] = ry;
            self.size[ry] += self.size[rx];
        } else {
            self.parent[ry] = rx;
            self.size[rx] += self.size[ry];
        }
        true
    }

    /// Return the size of the component containing `x`.
    ///
    /// ### Params
    ///
    /// * `x` - Element to look up
    ///
    /// ### Returns
    ///
    /// Number of elements in the component
    pub fn component_size(&mut self, x: usize) -> usize {
        let r = self.find(x);
        self.size[r]
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disjoint_set_basic() {
        let mut ds = DisjointSet::new(5);

        assert!(!ds.connected(0, 1));
        assert!(ds.union(0, 1));
        assert!(ds.connected(0, 1));
        assert!(!ds.union(0, 1)); // already connected
    }

    #[test]
    fn test_disjoint_set_chain() {
        let mut ds = DisjointSet::new(5);
        ds.union(0, 1);
        ds.union(1, 2);
        ds.union(2, 3);
        ds.union(3, 4);

        for i in 0..5 {
            for j in 0..5 {
                assert!(ds.connected(i, j));
            }
        }
    }

    #[test]
    fn test_disjoint_set_two_components() {
        let mut ds = DisjointSet::new(6);
        ds.union(0, 1);
        ds.union(1, 2);
        ds.union(3, 4);
        ds.union(4, 5);

        assert!(ds.connected(0, 2));
        assert!(ds.connected(3, 5));
        assert!(!ds.connected(0, 3));
    }

    #[test]
    fn test_sized_disjoint_set() {
        let mut ds = SizedDisjointSet::new(5);

        assert_eq!(ds.component_size(0), 1);
        ds.union(0, 1);
        assert_eq!(ds.component_size(0), 2);
        assert_eq!(ds.component_size(1), 2);
        ds.union(0, 2);
        assert_eq!(ds.component_size(2), 3);
    }

    #[test]
    fn test_sized_disjoint_set_full_merge() {
        let mut ds = SizedDisjointSet::new(4);
        ds.union(0, 1);
        ds.union(2, 3);
        ds.union(0, 3);
        assert_eq!(ds.component_size(0), 4);
        assert_eq!(ds.component_size(3), 4);
    }
}
