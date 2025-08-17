use log::debug;
use nalgebra::{DMatrix, DVector};
use rand::distributions::weighted::WeightedIndex;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::alignment::Sequences;
use crate::evolutionary_distances::{EvolutionaryDistance, LevenshteinDNACorrected};
use crate::tree::nj_matrices::NJMat;
use crate::tree::tree_builder::TreeBuilder;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub enum Randomise {
    Deterministic,
    Temperature(f64),
}

pub struct NJTreeBuilder<D: EvolutionaryDistance> {
    temperature: Randomise,
    distance_function: D,
}

impl<D: EvolutionaryDistance> TreeBuilder for NJTreeBuilder<D> {
    fn build(&self, sequences: &Sequences) -> Result<Tree> {
        let nj_data = self.compute_distance_matrix(sequences);
        self.build_nj_tree_from_matrix(nj_data, sequences)
    }
}

impl<D: EvolutionaryDistance + Default> Default for NJTreeBuilder<D> {
    fn default() -> Self {
        NJTreeBuilder {
            temperature: Randomise::Deterministic,
            distance_function: D::default(),
        }
    }
}

impl<D: EvolutionaryDistance> NJTreeBuilder<D> {
    pub fn new(temperature: Randomise, distance_function: D) -> Self {
        let temperature = match temperature {
            Randomise::Temperature(t) if t < 0.0 => {
                debug!("Temperature shouldn't be less than 0.00, setting to 0.00");
                Randomise::Temperature(0.0)
            }
            Randomise::Temperature(t) if t > 1.0 => {
                debug!("Temperature shouldn't be greater than 1.00, setting to 1.00");
                Randomise::Temperature(1.0)
            }
            _ => temperature,
        };
        Self {
            temperature,
            distance_function,
        }
    }

    fn index_to_ij_rowwise_nodiag(k: usize) -> (usize, usize) {
        // 0 indexed
        let p = ((1 + 8 * k).isqrt() - 1) / 2;
        let i = p + 1;
        let j = k - p * (p + 1) / 2;
        (i, j)
    }

    fn softmax_vector_mut(mut v: DVector<f64>) -> DVector<f64> {
        v = v.map(|i| i.exp());
        let sum_j = v.sum();
        v.unscale(sum_j)
    }

    fn temperature_branch(delta_tree_len: DVector<f64>, temperature: f64) -> (usize, usize) {
        debug_assert!(
            !delta_tree_len.is_empty(),
            "The input vector must not be empty."
        );
        if delta_tree_len.len() == 1 {
            return NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(0)
        }
        // Based on nj_correct test case, negative distances are common in compute_nj_q, which should be multiplied by -1 for small (desireable) distances to be larger in comparison to large distances. Allows for use of softmax
        let scaled_mat = delta_tree_len.scale(-1.0);
        // I think this messes up scaled_mat because of function implementation, borrows and never gives it back, could be reworked I think?
        let mut exp_mat = NJTreeBuilder::<LevenshteinDNACorrected>::softmax_vector_mut(scaled_mat);
        debug!("Probability Sum: {}", exp_mat.sum());
        // Use temperature with uniform value to calculate probability
        let uniform: f64 =
            1.0 / (((delta_tree_len.nrows().pow(2) - delta_tree_len.nrows()) / 2) as f64);
        // Temperature probabilities, temp=0.0 means uniform, temp=1.0 means softmax of distances
        exp_mat = exp_mat.map(|i| (temperature * uniform) + ((1.0 - temperature) * i));
        // This is our criterion for choosing, can use random seed because iteration is deterministic
        let dist = WeightedIndex::new(exp_mat.data.as_vec().iter()).unwrap();
        let mut rng = StdRng::from_seed([0; 32]);
        NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(dist.sample(&mut rng))
    }

    // Used with Randomise::Deterministic
    fn argmin_wo_diagonal(q: DVector<f64>) -> (usize, usize) {
        debug_assert!(!q.is_empty(), "The input matrix must not be empty.");
        if q.nrows() == 1 {
            return (1,0);
        }
        let mut arg_min = vec![];
        let mut val_min = &f64::MAX;
        for i in 0..q.nrows() {
            let val = &q[i];
            if val < val_min {
                val_min = val;
                // Convert usize to u32?
                arg_min =
                    vec![NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(i)];
            } else if val == val_min {
                arg_min
                    .push(NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(i));
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
        for cur_idx in n..=root_idx {
            let q = nj_data.compute_nj_delta_tree_length();
            let (i, j) = match self.temperature {
                Randomise::Temperature(t) => Self::temperature_branch(q, t),
                Randomise::Deterministic => {
                    NJTreeBuilder::<LevenshteinDNACorrected>::argmin_wo_diagonal(q)
                }
            };
            println!("{i}, {j}");
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

    fn compute_distance_matrix(&self, sequences: &Sequences) -> NJMat {
        let nseqs = sequences.len();
        let mut distances = DMatrix::zeros(nseqs, nseqs);
        for i in 0..nseqs {
            for j in (i + 1)..nseqs {
                let corrected_dist = self
                    .distance_function
                    .dist(sequences.record(i), sequences.record(j));
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
    use crate::{record_wo_desc as record, tree};
    use nalgebra::{dmatrix, dvector, DVector};

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
        //Changed test for empty vector, since argmin with 1 vector length will return first branch as delta_tree_length vector is different
        NJTreeBuilder::<LevenshteinDNACorrected>::argmin_wo_diagonal(DVector::<f64>::from_vec(
            vec![],
        ));
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
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
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
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
        let mat = nj_builder.compute_distance_matrix(&sequences);
        let true_mat = dmatrix![
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.0, 0.0, 0.2326161962278796, 0.051744653615213576;
        0.2326161962278796, 0.2326161962278796, 0.0, 0.28924686060898847;
        0.051744653615213576, 0.051744653615213576, 0.28924686060898847, 0.0];
        assert_eq!(mat.distances, true_mat);
    }

    #[test]
    fn nj_tree_original_paper() {
        // Compare against the original paper tree
        // https://academic.oup.com/mbe/article/4/4/406/1029664
        let nj_distances = NJMat {
            idx: (0..8).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 7.0, 8.0, 11.0, 13.0, 16.0, 13.0, 17.0;
                7.0, 0.0, 5.0, 8.0, 10.0, 13.0, 10.0, 14.0;
                8.0, 5.0, 0.0, 5.0, 7.0, 10.0, 7.0, 11.0;
                11.0, 8.0, 5.0, 0.0, 8.0, 11.0, 8.0, 12.0;
                13.0, 10.0, 7.0, 8.0, 0.0, 5.0, 6.0, 10.0;
                16.0, 13.0, 10.0, 11.0, 5.0, 0.0, 9.0, 13.0;
                13.0, 10.0, 7.0, 8.0, 6.0, 9.0, 0.0, 8.0;
                17.0, 14.0, 11.0, 12.0, 10.0, 13.0, 8.0, 0.0;
            ],
        };
        let sequences = Sequences::new((1..=8).map(|i| record!(&i.to_string(), b"")).collect());
        let nj_tree = NJTreeBuilder::<LevenshteinDNACorrected>::default()
            .build_nj_tree_from_matrix(nj_distances, &sequences)
            .unwrap();
        let correct_tree =
            tree!("((8:6,7:2):0.5,((5:1,6:4):2,(4:3,(3:1,(1:5,2:2):2):1):2):0.5):0.0;");
        assert_eq!(nj_tree.length, correct_tree.length);
        for leaf in nj_tree.leaves() {
            assert_eq!(leaf.blen, correct_tree.by_id(&leaf.id).blen);
        }
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
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
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
        // Illegal for this test since it's DNA corrected distance
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
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
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
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
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
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
        let nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
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
        let _nj_builder: NJTreeBuilder<LevenshteinDNACorrected> = NJTreeBuilder::default();
        let _nj_builder = NJTreeBuilder::new(Randomise::Deterministic, LevenshteinDNACorrected {});
        // Have to wrap distance function in Some, because otherwise Into has to be implemented for DistanceFunction type
        let _nj_builder =
            NJTreeBuilder::new(Randomise::Temperature(0.0), LevenshteinDNACorrected {});
        let _nj_builder =
            NJTreeBuilder::new(Randomise::Temperature(1.0), LevenshteinDNACorrected {});
    }

    #[test]
    fn lower_triangle_index_conversion() {
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(0),
            (1, 0)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(1),
            (2, 0)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(2),
            (2, 1)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(3),
            (3, 0)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected>::index_to_ij_rowwise_nodiag(5),
            (3, 2)
        );
    }

    #[test]
    fn compute_nj_delta_tree_length_vector() {
        let nj_distances = NJMat {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        }; 
        let q = nj_distances.compute_nj_delta_tree_length();
        assert_eq!(q, dvector![-50.0,-38.0,-38.0,-34.0,-34.0,-40.0,-34.0,-34.0,-40.0,-48.0])
    }

    #[test]
    fn argmin_wo_diagonal_vector() {
        let delta_tree_length = dvector![-50.0,-38.0,-38.0,-34.0,-34.0,-40.0,-34.0,-34.0,-40.0,-48.0];
        assert_eq!(NJTreeBuilder::<LevenshteinDNACorrected>::argmin_wo_diagonal(delta_tree_length), (1,0));
    }

    #[test]
    fn softmax_vector_mut_test() {
        let delta_tree_length = dvector![1.3,5.1,2.2,0.7,1.1];
        let softmax_vector = NJTreeBuilder::<LevenshteinDNACorrected>::softmax_vector_mut(delta_tree_length);
        println!("{softmax_vector:?}");
        assert_eq!(softmax_vector, dvector![0.020190464732580685, 0.9025376890165726, 0.04966052987196013, 0.011080761983386346, 0.01653055439550022]);
        assert_eq!(softmax_vector.sum(), 1.0);
    }
    
    // Rethink adding these tests, would need random seed to work
    // #[test]
    // fn nj_builder_uniform() {
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
    //     let nj_builder = NJBuilder::new(Randomise::Temperature(0.0), LevenshteinDNACorrected {});
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

    #[test]
    fn nj_builder_softmax() {
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
        let nj_builder = NJTreeBuilder::new(Randomise::Temperature(0.0), LevenshteinDNACorrected);
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
}
