use crate::tree::{
    nj_matrices::{Mat, NJMat},
    NodeIdx, Tree,
};

use crate::alignment::Sequences;
use crate::Result;
use bio::alignment::distance::levenshtein;
use nalgebra::{max, DMatrix};

pub struct NJBuilder {
    use_random: f64,
}

// Functions are public for now, let me know if they should be private and reimplement the tests
impl NJBuilder {
    //Copied over the NJ_builder functionality from mod.rs
    pub(crate) fn new(use_random: f64) -> Result<Self> {
        Ok(Self {
            use_random: use_random,
        })
    }

    pub fn argmin_wo_diagonal(q: Mat) -> (usize, usize) {
        debug_assert!(!q.is_empty(), "The input matrix must not be empty.");
        debug_assert!(
            q.ncols() > 1 && q.nrows() > 1,
            "The input matrix should have more than 1 element."
        );
        let mut arg_min = vec![];
        let mut val_min = &f64::MAX;
        for i in 0..q.nrows() {
            for j in 0..i {
                let val = &q[(i, j)];
                if val < val_min {
                    val_min = val;
                    arg_min = vec![(i, j)];
                } else if val == val_min {
                    arg_min.push((i, j));
                }
            }
        }

        cfg_if::cfg_if! {
        if #[cfg(feature = "deterministic")]{
            arg_min[0]
        } else {
            arg_min[Self::rng_len(arg_min.len())]
        }
        }
    }

    #[cfg(not(feature = "deterministic"))]
    fn rng_len(l: usize) -> usize {
        rand::random::<usize>() % l
    }

    pub fn build_nj_tree_from_matrix(mut nj_data: NJMat, sequences: &Sequences) -> Result<Tree> {
        let n = nj_data.distances.ncols();
        let mut tree = Tree::new(sequences)?;
        let root_idx = usize::from(&tree.root);
        for cur_idx in n..=root_idx {
            let q = nj_data.compute_nj_q();
            let (i, j) = NJBuilder::argmin_wo_diagonal(q);
            let idx_new = cur_idx;
            let (blen_i, blen_j) = nj_data.branch_lengths(i, j, cur_idx == root_idx);
            tree.add_parent(idx_new, &nj_data.idx[i], &nj_data.idx[j], blen_i, blen_j);
            nj_data = nj_data
                .add_merge_node(idx_new)
                .recompute_new_node_distances(i, j)
                .remove_merged_nodes(i, j);
        }
        tree.n = n;
        tree.complete = true;
        tree.compute_postorder();
        tree.compute_preorder();
        tree.height = tree.nodes.iter().map(|node| node.blen).sum();
        Ok(tree)
    }
    //Converted this to method instead of associated function, we can decide which to use
    pub fn build_nj_tree(&self, sequences: &Sequences) -> Result<Tree> {
        //Convert this to self if using methods also add distance function argument
        let nj_data = NJBuilder::compute_distance_matrix(sequences);
        NJBuilder::build_nj_tree_from_matrix(nj_data, sequences)
    }

    pub fn compute_distance_matrix(sequences: &Sequences) -> NJMat {
        let nseqs = sequences.len();
        let mut distances = DMatrix::zeros(nseqs, nseqs);
        for i in 0..nseqs {
            for j in (i + 1)..nseqs {
                let seq_i = sequences.record(i).seq();
                let seq_j = sequences.record(j).seq();
                let lev_dist = levenshtein(seq_i, seq_j) as f64;
                let proportion_diff = f64::min(
                    lev_dist / (max(seq_i.len(), seq_j.len()) as f64),
                    0.75 - f64::EPSILON,
                );
                let corrected_dist = -3.0 / 4.0 * (1.0 - 4.0 / 3.0 * proportion_diff).ln();
                distances[(i, j)] = corrected_dist;
                distances[(j, i)] = corrected_dist;
            }
        }
        NJMat {
            idx: (0..nseqs).map(NodeIdx::Leaf).collect(),
            distances,
        }
    }
}
// Implement the tests at the bottom of this module for ability to use private functions
