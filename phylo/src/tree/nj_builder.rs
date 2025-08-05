use std::default;

use crate::tree::{
    nj_matrices::{Mat, NJMat},
    NodeIdx, Tree,
};

use crate::alignment::Sequences;
use crate::tree::tree_builder::TreeBuilder;
use crate::Result;
use bio::alignment::distance::levenshtein;
use log::debug;
use nalgebra::{max, DMatrix};

// Made this into a type so that it can be changed easily if TextSlice changes, might be a better solution out there
type DistanceFunction = fn(&[u8], &[u8]) -> u32;

pub struct NJBuilder {
    temperature: Option<f64>,
    distance_function: DistanceFunction, //This is the same type as TextSlice for bio, just explicitly defines since TextSlice is private, Option
}

impl TreeBuilder for NJBuilder {
    fn build_tree(&self, sequences: &Sequences) -> Result<Tree> {
        let nj_data = self.compute_distance_matrix(sequences);
        // Now returns from a method rather than associated function, for use of temperature instantiated value
        self.build_nj_tree_from_matrix(nj_data, sequences)
    }
}

impl default::Default for NJBuilder {
    fn default() -> Self {
        NJBuilder {
            temperature: None,
            distance_function: levenshtein,
        }
    }
}

impl NJBuilder {
    // With the way we are using Option makes sense to use Into to allow for passing
    pub(crate) fn new<T>(temperature: T, distance_function: Option<DistanceFunction>) -> Self
    where
        T: Into<Option<f64>>,
    {
        // Add checking of unwrap for default when not None is passed in? Warning to user
        // None temperature should be allowed, or if float, add stochastic
        let temperature = temperature.into();
        if let Some(t) = temperature {
            assert!(
                (0.0..=1.0).contains(&t),
                "Temperature must be between 0.0 and 1.0 inclusive"
            );
        }
        // Default is levenshtein
        let distance_function = distance_function.unwrap_or(levenshtein);
        Self {
            temperature,
            distance_function,
        }
    }

    fn softmax_branch(&self, q: Mat) -> (usize, usize) {
        debug_assert!(!q.is_empty(), "The input matrix must not be empty.");
        debug_assert!(
            q.ncols() > 1 && q.nrows() > 1,
            "The input matrix should have more than 1 element."
        );
        debug_assert!(q.ncols() == q.nrows(), "The input matrix should be square.");
        // Test better outputs of computer_nj_q() as this affects this...
        // Based on nj_correct test case, negative distances are here. Otherwise, multiplying by negative 1 should correctly make large distances very small, and small distances, still small, or large if negative
        let mut exp_mat = q.scale(-1.0);
        // e^i on all elements of matrix
        exp_mat = exp_mat.lower_triangle().map(|i| i.exp());
        // With large matrices/numbers there is a slight discrepancy due to floating point numbers, greater than 1
        let exp_sum = exp_mat.lower_triangle().sum() - exp_mat.diagonal().sum();
        // This is softmax, on each value e^i/sum(i..n)(e^i)
        exp_mat = exp_mat.unscale(exp_sum);
        // This can occasionally be greater than 1 due to handling of large numbers and floating points
        debug!(
            "Probability L Triangle Sum: {}",
            exp_mat.lower_triangle().sum() - exp_mat.diagonal().sum()
        );

        // Use temperature with uniform value to calculate probability
        let uniform: f64 = 1.0 / (((q.nrows().pow(2) - q.nrows()) / 2) as f64);
        // Temperature probabilities, temp=0.0 means uniform, temp=1.0 means softmax of distances
        exp_mat = exp_mat.lower_triangle().map(|i| {
            ((1.0 - self.temperature.unwrap()) * uniform) + (self.temperature.unwrap() * i)
        });

        // This is our criterion for choosing, can use random seed because iteration is deterministic
        let branch_choice: f64 = rand::random();
        let mut prob_sum = 0.0;
        for i in 1..q.nrows() {
            for j in 0..i {
                let val = exp_mat[(i, j)];
                prob_sum += val;
                if prob_sum > branch_choice {
                    return (i, j);
                }
            }
        }
        // Change? default, warning, fallback to argmin, this should never happen based on probability contraints.
        panic!("No return from softmax branch");
    }

    // This is the original implementation, to be used with temperature None
    fn argmin_wo_diagonal(q: Mat) -> (usize, usize) {
        debug_assert!(!q.is_empty(), "The input matrix must not be empty.");
        debug_assert!(
            q.ncols() > 1 && q.nrows() > 1,
            "The input matrix should have more than 1 element."
        );
        debug_assert!(q.ncols() == q.nrows(), "The input matrix should be square.");
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

    fn build_nj_tree_from_matrix(&self, mut nj_data: NJMat, sequences: &Sequences) -> Result<Tree> {
        let n = nj_data.distances.ncols();
        let mut tree = Tree::new(sequences)?;
        let root_idx = usize::from(&tree.root);
        //Wrapped in temperature check for now, can implement stochastic better in future
        for cur_idx in n..=root_idx {
            let q = nj_data.compute_nj_q();
            // This is where we choose our next branch, if temperature exists softmax, if not argmin, default behavior
            let (i, j) = match self.temperature {
                Some(_) => self.softmax_branch(q),
                _ => NJBuilder::argmin_wo_diagonal(q),
            };
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
        tree.length = tree.nodes.iter().map(|node| node.blen).sum();
        Ok(tree)
    }
    //Converted this to method instead of associated function, we can decide which to use

    fn compute_distance_matrix(&self, sequences: &Sequences) -> NJMat {
        let nseqs = sequences.len();
        let mut distances = DMatrix::zeros(nseqs, nseqs);
        for i in 0..nseqs {
            for j in (i + 1)..nseqs {
                let seq_i = sequences.record(i).seq();
                let seq_j = sequences.record(j).seq();
                let dist = (self.distance_function)(seq_i, seq_j) as f64;
                let proportion_diff = f64::min(
                    dist / (max(seq_i.len(), seq_j.len()) as f64),
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
#[cfg(test)]
mod private_tests {
    //From test.rs in tree, so we can use macros
    use crate::record_wo_desc as record;
    use nalgebra::{dmatrix, DMatrix};

    use super::*;
    use crate::tree::{
        Node,
        NodeIdx::{self, Internal as I, Leaf as L},
    };

    #[cfg(test)]
    fn is_unique<T: std::cmp::Eq + std::hash::Hash>(vec: &[T]) -> bool {
        let set: std::collections::HashSet<_> = vec.iter().collect();
        set.len() == vec.len()
    }

    #[test]
    #[should_panic]
    fn test_argmin_fail() {
        //Instantiate NJBuilder instance every time
        NJBuilder::argmin_wo_diagonal(DMatrix::<f64>::from_vec(1, 1, vec![0.0]));
    }

    #[test]
    fn compute_distance_matrix_close() {
        let sequences = Sequences::new(vec![
            record!("A0", b"C"),
            record!("B1", b"A"),
            record!("C2", b"AA"),
            record!("D3", b"A"),
            record!("E4", b"CC"),
        ]);
        //For now, may instantiate NJBuilder instance every time
        let nj_builder = NJBuilder::default();
        let mat = nj_builder.compute_distance_matrix(&sequences);
        let true_mat = dmatrix![
        0.0, 26.728641210756745, 26.728641210756745, 26.728641210756745, 0.8239592165010822;
        26.728641210756745, 0.0, 0.8239592165010822, 0.0, 26.728641210756745;
        26.728641210756745, 0.8239592165010822, 0.0, 0.8239592165010822, 26.728641210756745;
        26.728641210756745, 0.0, 0.8239592165010822, 0.0, 26.728641210756745;
        0.8239592165010822, 26.728641210756745, 26.728641210756745, 26.728641210756745, 0.0];
        assert_eq!(mat.distances, true_mat);
    }

    #[test]
    fn compute_distance_matrix_far() {
        let sequences = Sequences::new(vec![
            record!("A0", b"AAAAAAAAAAAAAAAAAAAA"),
            record!("B1", b"AAAAAAAAAAAAAAAAAAAA"),
            record!("C2", b"AAAAAAAAAAAAAAAAAAAAAAAAA"),
            record!("D3", b"CAAAAAAAAAAAAAAAAAAA"),
        ]);
        //For now, may instantiate NJBuilder instance every time
        let nj_builder = NJBuilder::default();
        let mat = nj_builder.compute_distance_matrix(&sequences);
        let true_mat = dmatrix![
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.2326161962278796, 0.2326161962278796, 0.0, 0.28924686060898847;
        0.051744653615213576, 0.051744653615213576, 0.28924686060898847, 0.0];
        assert_eq!(mat.distances, true_mat);
    }

    #[test]
    fn nj_correct_2() {
        // NJ based on example from https://www.tenderisthebyte.com/blog/2022/08/31/neighbor-joining-trees/#neighbor-joining-trees
        let nj_distances = NJMat {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 4.0, 5.0, 10.0;
                4.0, 0.0, 7.0, 12.0;
                5.0, 7.0, 0.0, 9.0;
                10.0, 12.0, 9.0, 0.0],
        };
        let sequences = Sequences::new(vec![
            record!("A", b""),
            record!("B", b""),
            record!("C", b""),
            record!("D", b""),
        ]);
        // Instantiate NJBuilder instance every time
        let nj_builder = NJBuilder::default();
        let tree = nj_builder
            .build_nj_tree_from_matrix(nj_distances, &sequences)
            .unwrap();
        assert_eq!(tree.by_id("A").blen, 1.0);
        assert_eq!(tree.by_id("B").blen, 3.0);
        assert_eq!(tree.by_id("C").blen, 2.0);
        assert_eq!(tree.by_id("D").blen, 7.0);
        assert_eq!(tree.node(&I(4)).blen, 1.0);
        assert_eq!(tree.node(&I(5)).blen, 1.0);
        assert_eq!(tree.len(), 7);
        assert_eq!(tree.postorder.len(), 7);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 7);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn protein_nj_correct() {
        // NJ based on example sequences from "./data/sequences_protein1.fasta"
        let nj_distances = NJMat {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.2, 0.2, 0.2, 0.0],
        };
        let sequences = Sequences::new(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
        ]);
        // Instantiate NJBuilder instance every time
        let nj_builder = NJBuilder::default();
        let tree = nj_builder
            .build_nj_tree_from_matrix(nj_distances, &sequences)
            .unwrap();
        assert_eq!(tree.len(), 7);
        assert_eq!(tree.postorder.len(), 7);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 7);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn nj_correct_wiki_example() {
        // NJ based on example from https://en.wikipedia.org/wiki/Neighbor_joining
        let nj_distances = NJMat {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let sequences = Sequences::new(vec![
            record!("a", b""),
            record!("b", b""),
            record!("c", b""),
            record!("d", b""),
            record!("e", b""),
        ]);
        let nj_builder = NJBuilder::default();
        let tree = nj_builder
            .build_nj_tree_from_matrix(nj_distances, &sequences)
            .unwrap();
        assert_eq!(tree.by_id("a").blen, 2.0);
        assert_eq!(tree.by_id("b").blen, 3.0);
        assert_eq!(tree.by_id("c").blen, 4.0);
        assert_eq!(tree.by_id("d").blen, 2.0);
        assert_eq!(tree.by_id("e").blen, 1.0);
        assert_eq!(tree.node(&I(5)).blen, 3.0);
        assert_eq!(tree.node(&I(6)).blen, 1.0);
        assert_eq!(tree.node(&I(7)).blen, 1.0);
        assert_eq!(tree.len(), 9);
        assert_eq!(tree.postorder.len(), 9);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 9);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn nj_correct() {
        let nj_distances = NJMat {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let sequences = Sequences::new(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
            record!("E4", b""),
        ]);
        let nj_builder = NJBuilder::default();
        let nj_tree = nj_builder
            .build_nj_tree_from_matrix(nj_distances, &sequences)
            .unwrap();
        let nodes = vec![
            Node::new_leaf(0, Some(I(5)), 2.0, "A0".to_string()),
            Node::new_leaf(1, Some(I(5)), 3.0, "B1".to_string()),
            Node::new_leaf(2, Some(I(7)), 4.0, "C2".to_string()),
            Node::new_leaf(3, Some(I(6)), 2.0, "D3".to_string()),
            Node::new_leaf(4, Some(I(6)), 1.0, "E4".to_string()),
            Node::new_internal(5, Some(I(7)), vec![L(1), L(0)], 3.0, "".to_string()),
            Node::new_internal(6, Some(I(8)), vec![L(4), L(3)], 1.0, "".to_string()),
            Node::new_internal(7, Some(I(8)), vec![I(5), L(2)], 1.0, "".to_string()),
            Node::new_internal(8, None, vec![I(7), I(6)], 0.0, "".to_string()),
        ];
        assert_eq!(nj_tree.root, I(8));
        assert_eq!(nj_tree.nodes, nodes);
    }

    #[test]
    fn nj_correct_web_example() {
        let nj_distances = NJMat {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                    0.0, 4.0, 5.0, 10.0;
                    4.0, 0.0, 7.0, 12.0;
                    5.0, 7.0, 0.0, 9.0;
                    10.0, 12.0, 9.0, 0.0],
        };
        let sequences = Sequences::new(vec![
            record!("A0", b""),
            record!("B1", b""),
            record!("C2", b""),
            record!("D3", b""),
        ]);
        let nj_builder = NJBuilder::default();
        let nj_tree = nj_builder
            .build_nj_tree_from_matrix(nj_distances, &sequences)
            .unwrap();
        let nodes = vec![
            Node::new_leaf(0, Some(I(4)), 1.0, "A0".to_string()),
            Node::new_leaf(1, Some(I(4)), 3.0, "B1".to_string()),
            Node::new_leaf(2, Some(I(5)), 2.0, "C2".to_string()),
            Node::new_leaf(3, Some(I(5)), 7.0, "D3".to_string()),
            Node::new_internal(4, Some(I(6)), vec![L(0), L(1)], 1.0, "".to_string()),
            Node::new_internal(5, Some(I(6)), vec![L(3), L(2)], 1.0, "".to_string()),
            Node::new_internal(6, None, vec![I(4), I(5)], 0.0, "".to_string()),
        ];

        assert_eq!(nj_tree.root, I(6));
        assert_eq!(nj_tree.nodes, nodes);
    }

    #[test]
    fn nj_builder_correct_creation() {
        let _nj_builder = NJBuilder::default();
        let _nj_builder = NJBuilder::new(None, None);
        // Have to wrap distance function in Some, because otherwise Into has to be implemented for DistanceFunction type
        let _nj_builder = NJBuilder::new(0.00, Some(levenshtein));
        let _nj_builder = NJBuilder::new(1.00, None);
    }

    #[test]
    #[should_panic]
    fn nj_builder_panic_creation() {
        //temperature over 1.0
        let _nj_builder = NJBuilder::new(2.0, None);
    }

    // Rethink adding these tests, would need random seed to work
    // #[test]
    // fn nj_builder_uniform() {
    //     // Stolen from nj_correct test
    //     let nj_distances = NJMat {
    //         idx: (0..4).map(NodeIdx::Leaf).collect(),
    //         distances: dmatrix![
    //                 0.0, 4.0, 5.0, 10.0;
    //                 4.0, 0.0, 7.0, 12.0;
    //                 5.0, 7.0, 0.0, 9.0;
    //                 10.0, 12.0, 9.0, 0.0],
    //     };
    //     let sequences = Sequences::new(vec![
    //         record!("A0", b""),
    //         record!("B1", b""),
    //         record!("C2", b""),
    //         record!("D3", b""),
    //     ]);
    //     let nj_builder = NJBuilder::new(0.0, None);
    //     let nj_tree = nj_builder
    //         .build_nj_tree_from_matrix(nj_distances, &sequences)
    //         .unwrap();
    //     let nodes = vec![
    //         Node::new_leaf(0, Some(I(4)), 1.0, "A0".to_string()),
    //         Node::new_leaf(1, Some(I(4)), 3.0, "B1".to_string()),
    //         Node::new_leaf(2, Some(I(5)), 2.0, "C2".to_string()),
    //         Node::new_leaf(3, Some(I(5)), 7.0, "D3".to_string()),
    //         Node::new_internal(4, Some(I(6)), vec![L(0), L(1)], 1.0, "".to_string()),
    //         Node::new_internal(5, Some(I(6)), vec![L(3), L(2)], 1.0, "".to_string()),
    //         Node::new_internal(6, None, vec![I(4), I(5)], 0.0, "".to_string()),
    //     ];
    //     assert_eq!(nj_tree.root, I(6));
    //     assert_eq!(nj_tree.nodes, nodes);
    // }

    // #[test]
    // fn nj_builder_softmax() {
    //     // Stolen from nj_correct test
    //     let nj_distances = NJMat {
    //         idx: (0..4).map(NodeIdx::Leaf).collect(),
    //         distances: dmatrix![
    //                 0.0, 4.0, 5.0, 10.0;
    //                 4.0, 0.0, 7.0, 12.0;
    //                 5.0, 7.0, 0.0, 9.0;
    //                 10.0, 12.0, 9.0, 0.0],
    //     };
    //     let sequences = Sequences::new(vec![
    //         record!("A0", b""),
    //         record!("B1", b""),
    //         record!("C2", b""),
    //         record!("D3", b""),
    //     ]);
    //     let nj_builder = NJBuilder::new(1.0, None);
    //     let nj_tree = nj_builder
    //         .build_nj_tree_from_matrix(nj_distances, &sequences)
    //         .unwrap();
    //     let nodes = vec![
    //         Node::new_leaf(0, Some(I(4)), 1.0, "A0".to_string()),
    //         Node::new_leaf(1, Some(I(4)), 3.0, "B1".to_string()),
    //         Node::new_leaf(2, Some(I(5)), 2.0, "C2".to_string()),
    //         Node::new_leaf(3, Some(I(5)), 7.0, "D3".to_string()),
    //         Node::new_internal(4, Some(I(6)), vec![L(0), L(1)], 1.0, "".to_string()),
    //         Node::new_internal(5, Some(I(6)), vec![L(3), L(2)], 1.0, "".to_string()),
    //         Node::new_internal(6, None, vec![I(4), I(5)], 0.0, "".to_string()),
    //     ];
    //     assert_eq!(nj_tree.root, I(6));
    //     assert_eq!(nj_tree.nodes, nodes);
    // }
}
