use crate::tree::NodeIdx::{self, Internal as Int};
use nalgebra::{max, min, DMatrix};
use std::fmt::{Display, Formatter, Result};

pub(super) type Mat = DMatrix<f64>;

#[derive(Debug)]
pub(super) struct NJMat {
    pub(super) idx: Vec<NodeIdx>,
    pub(super) distances: Mat,
}

impl Display for NJMat {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{:?}\n {}", self.idx, self.distances)
    }
}

impl NJMat {
    pub(super) fn add_merge_node(mut self, idx_new: usize) -> Self {
        let new_row_index = self.distances.nrows();
        self.distances = self
            .distances
            .insert_row(new_row_index, 0.0)
            .insert_column(new_row_index, 0.0);
        self.idx.push(Int(idx_new));
        self
    }

    pub(super) fn remove_merged_nodes(mut self, idx_i: usize, idx_j: usize) -> Self {
        self.distances = self
            .distances
            .remove_columns_at(&[idx_i, idx_j])
            .remove_rows_at(&[idx_i, idx_j]);
        self.idx.remove(max(idx_i, idx_j));
        self.idx.remove(min(idx_i, idx_j));
        debug_assert!(idx_i > idx_j);
        self
    }

    pub(super) fn recompute_new_node_distances(mut self, i: usize, j: usize) -> Self {
        let cur_n_nodes = self.distances.ncols() - 1;
        for k in (0..cur_n_nodes).filter(|&k| k != i && k != j) {
            let new_dist =
                (self.distances[(i, k)] + self.distances[(j, k)] - self.distances[(i, j)]) / 2.0;
            self.distances[(cur_n_nodes, k)] = new_dist;
            self.distances[(k, cur_n_nodes)] = new_dist;
        }
        self
    }

    pub(super) fn branch_lengths(&self, i: usize, j: usize, is_root: bool) -> (f64, f64) {
        let blen_i = if is_root {
            self.distances[(i, j)] / 2.0
        } else {
            self.distances[(i, j)] / 2.0
                + (self.distances.row_sum()[i] - self.distances.row_sum()[j])
                    / (2 * (self.distances.ncols() - 2)) as f64
        };
        let blen_j = if is_root {
            self.distances[(j, i)] / 2.0
        } else {
            self.distances[(j, i)] / 2.0
                + (self.distances.row_sum()[j] - self.distances.row_sum()[i])
                    / (2 * (self.distances.ncols() - 2)) as f64
        };
        (
            if blen_i <= 0.0 { f64::EPSILON } else { blen_i },
            if blen_j <= 0.0 { f64::EPSILON } else { blen_j },
        )
    }

    pub(super) fn compute_nj_q(&self) -> Mat {
        let n = self.distances.ncols();
        let s = self.distances.row_sum();
        Mat::from_fn(n, n, |r, c| -> f64 {
            if r == c {
                0.0
            } else {
                (n - 2) as f64 * self.distances[(r, c)] - s[r] - s[c]
            }
        })
    }
}
