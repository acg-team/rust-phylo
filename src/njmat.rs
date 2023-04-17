use nalgebra::{max, min, DMatrix};
use super::tree::NodeIdx;
use super::tree::NodeIdx::Internal as Int;

use std::fmt::Display;

pub(crate) type Mat = DMatrix<f32>;

#[derive(Debug)]
pub(crate) struct NJMat {
    pub(crate) idx: Vec<NodeIdx>,
    pub(crate) distances: Mat,
}

impl Display for NJMat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}\n {}", self.idx, self.distances)
    }
}

impl NJMat {
    pub(crate) fn add_merge_node(mut self, idx_new: usize) -> Self {
        let new_row_index = self.distances.nrows();
        self.distances = self
            .distances
            .insert_row(new_row_index, 0.0)
            .insert_column(new_row_index, 0.0);
        self.idx.push(Int(idx_new));
        self
    }

    pub(crate) fn remove_merged_nodes(mut self, idx_i: usize, idx_j: usize) -> Self {
        self.distances = self
            .distances
            .remove_columns_at(&[idx_i, idx_j])
            .remove_rows_at(&[idx_i, idx_j]);
        self.idx.remove(max(idx_i, idx_j));
        self.idx.remove(min(idx_i, idx_j));
        self
    }

    pub(crate) fn recompute_new_node_distances(mut self, i: usize, j: usize) -> Self {
        let cur_n_nodes = self.distances.ncols() - 1;
        for k in (0..cur_n_nodes).filter(|&k| k != i && k != j) {
            let new_dist =
                (self.distances[(i, k)] + self.distances[(j, k)] - self.distances[(i, j)]) / 2.0;
            self.distances[(cur_n_nodes, k)] = new_dist;
            self.distances[(k, cur_n_nodes)] = new_dist;
        }
        self
    }

    pub(crate) fn branch_lengths(&self, i: usize, j: usize, is_root: bool) -> (f32, f32) {
        let blen_i = if is_root {
            self.distances[(i, j)] / 2.0
        } else {
            self.distances[(i, j)] / 2.0
                + (self.distances.row_sum()[i] - self.distances.row_sum()[j])
                    / (2 * (self.distances.ncols() - 2)) as f32
        };
        let blen_j = self.distances[(i, j)] - blen_i;
        (blen_i, if blen_j < 0.0 {0.0} else {blen_j})
    }

    pub(crate) fn compute_nj_q(&self) -> Mat {
        let n = self.distances.ncols();
        let s = self.distances.row_sum();
        Mat::from_fn(n, n, |r, c| {
            if r == c {
                0.0
            } else {
                (n - 2) as f32 * self.distances[(r, c)] - s[r] - s[c]
            }
        })
    }
}
