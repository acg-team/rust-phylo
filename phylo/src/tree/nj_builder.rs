use log::debug;
use nalgebra::{DMatrix, DVector};
use rand::distributions::weighted::WeightedIndex;

use crate::alignment::Sequences;
use crate::evolutionary_distances::EvolutionaryDistance;
use crate::random::RandomSource;
use crate::tree::nj_matrices::DistanceMatrix;
use crate::tree::tree_builder::TreeBuilder;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub enum Strategy {
    Deterministic,
    SoftmaxUniform(f64),
}

pub struct NJTreeBuilder<'a, D: EvolutionaryDistance, R: RandomSource> {
    randomise: Strategy,
    distance_function: D,
    rng: &'a R,
}

impl<'a, D: EvolutionaryDistance, R: RandomSource> TreeBuilder for NJTreeBuilder<'a, D, R> {
    fn build(&self, sequences: &Sequences) -> Result<Tree> {
        let distances = self.compute_distance_matrix(sequences);
        self.build_from_distances(distances, sequences)
    }
}

impl<'a, D: EvolutionaryDistance, R: RandomSource> NJTreeBuilder<'a, D, R> {
    /// Creates a Neighbor Joining Tree Builder object with Deterministic strategy, which uses argmin to minimize the tree length
    pub fn new(distance_function: D, rng: &'a R) -> Self {
        Self {
            randomise: Strategy::Deterministic,
            distance_function,
            rng,
        }
    }

    /// Creates a Neighbor Joining Tree object with Softmax strategy, introduces stochasticity to tree building.
    /// Temperature can be between 0.0 and 1.0 and interpolates between the uniform and softmax distributions to select
    /// the next pair of nodes to join. A temperature of 0.0 is fully uniform, while a temperature of 1.0 is fully softmax.
    pub fn new_with_softmax(distance_function: D, rng: &'a R, temperature: f64) -> Self {
        if temperature > 1.0 {
            debug!("Temperature should not be greater than 1.0 (set to {temperature}), clamping to 1.0");
        } else if temperature < 0.0 {
            debug!(
                "Temperature should not be less than 0.0 (set to {temperature}), clamping to 0.0"
            );
        }
        let t = temperature.clamp(0.0, 1.0);
        Self {
            randomise: Strategy::SoftmaxUniform(t),
            distance_function,
            rng,
        }
    }

    fn lower_triangle_index(k: usize) -> (usize, usize) {
        // 0 indexed
        let p = ((1 + 8 * k).isqrt() - 1) / 2;
        let i = p + 1;
        let j = k - p * (p + 1) / 2;
        (i, j)
    }

    fn softmax(mut v: DVector<f64>) -> DVector<f64> {
        v = v.map(|i| i.exp());
        let sum_j = v.sum();
        v.unscale(sum_j)
    }

    fn softmax_uniform(
        delta_tree_len: DVector<f64>,
        temperature: f64,
        rng: &impl RandomSource,
    ) -> usize {
        debug_assert!(
            !delta_tree_len.is_empty(),
            "The input vector must not be empty."
        );
        if delta_tree_len.len() == 1 {
            return 0;
        }
        //Invert distances for softmax
        let inverted_delta = delta_tree_len.scale(-1.0);
        let mut exp_mat = NJTreeBuilder::<D, R>::softmax(inverted_delta);
        let uniform: f64 =
            1.0 / (((delta_tree_len.nrows().pow(2) - delta_tree_len.nrows()) / 2) as f64);
        // Interpolated probabilities, temp=0.0 means uniform, temp=1.0 means softmax of distances
        exp_mat = exp_mat.map(|i| (temperature * uniform) + ((1.0 - temperature) * i));
        let dist = WeightedIndex::new(exp_mat.data.as_vec().iter()).unwrap();
        rng.sample(&dist)
    }

        }
    }

    fn build_from_distances(
        &self,
        mut distances: DistanceMatrix,
        sequences: &Sequences,
    ) -> Result<Tree> {
        let n = distances.distances.ncols();
        let mut tree = Tree::new(sequences)?;
        let root_idx = usize::from(&tree.root);
        for cur_idx in n..=root_idx {
            let q = distances.delta_tree_length();
            let index = match self.randomise {
                Strategy::SoftmaxUniform(t) => Self::softmax_uniform(q, t, self.rng),
                Strategy::Deterministic => q.argmin().0,
            };
            let (i, j) = Self::lower_triangle_index(index);
            let idx_new = cur_idx;
            let (blen_i, blen_j) = distances.branch_lengths(i, j, cur_idx == root_idx);
            tree.add_parent(
                idx_new,
                &distances.idx[i],
                &distances.idx[j],
                blen_i,
                blen_j,
            );
            distances = distances
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

    fn compute_distance_matrix(&self, sequences: &Sequences) -> DistanceMatrix {
        let nseqs = sequences.len();
        let mut distances = DMatrix::zeros(nseqs, nseqs);
        for i in 0..nseqs {
            for j in (i + 1)..nseqs {
                let dist = self
                    .distance_function
                    .dist(sequences.record(i), sequences.record(j));
                distances[(i, j)] = dist;
                distances[(j, i)] = dist;
            }
        }
        DistanceMatrix {
            idx: (0..nseqs).map(NodeIdx::Leaf).collect(),
            distances,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    use crate::evolutionary_distances::LevenshteinDNACorrected;
    use crate::random::{DefaultGenerator, FakeGenerator};
    use crate::tree::Node;
    use crate::tree::NodeIdx::{self, Internal as I, Leaf as L};
    use crate::{record_wo_desc as record, tree};

    #[cfg(test)]
    fn is_unique<T: std::cmp::Eq + std::hash::Hash>(vec: &[T]) -> bool {
        let set: std::collections::HashSet<_> = vec.iter().collect();
        set.len() == vec.len()
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
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
        let nj_distances = DistanceMatrix {
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
        let rng = FakeGenerator::new();
        let nj_tree = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng)
            .build_from_distances(nj_distances, &sequences)
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
        let nj_distances = DistanceMatrix {
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
        let tree = nj_builder
            .build_from_distances(nj_distances, &sequences)
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
        let nj_distances = DistanceMatrix {
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
        let tree = nj_builder
            .build_from_distances(nj_distances, &sequences)
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
        let nj_distances = DistanceMatrix {
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
        let tree = nj_builder
            .build_from_distances(nj_distances, &sequences)
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
        let nj_distances = DistanceMatrix {
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
        let nj_tree = nj_builder
            .build_from_distances(nj_distances, &sequences)
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
        let nj_distances = DistanceMatrix {
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
        let rng = FakeGenerator::new();
        let nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
        let nj_tree = nj_builder
            .build_from_distances(nj_distances, &sequences)
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
        let rng = FakeGenerator::new();
        let _nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &rng);
        let _nj_builder = NJTreeBuilder::new(LevenshteinDNACorrected {}, &DefaultGenerator::new(0));
        let _nj_builder = NJTreeBuilder::new_with_softmax(
            LevenshteinDNACorrected {},
            &DefaultGenerator::new(0),
            0.0,
        );
        let _nj_builder = NJTreeBuilder::new_with_softmax(
            LevenshteinDNACorrected {},
            &DefaultGenerator::new(0),
            1.0,
        );
    }

    #[test]
    fn lower_triangle_index_conversion() {
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::lower_triangle_index(0),
            (1, 0)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::lower_triangle_index(1),
            (2, 0)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::lower_triangle_index(2),
            (2, 1)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::lower_triangle_index(3),
            (3, 0)
        );
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::lower_triangle_index(5),
            (3, 2)
        );
    }

    #[test]
    fn delta_tree_length() {
        let nj_distances = DistanceMatrix {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let q = nj_distances.delta_tree_length();
        assert_eq!(
            q,
            dvector![-50.0, -38.0, -38.0, -34.0, -34.0, -40.0, -34.0, -34.0, -40.0, -48.0]
        )
    }

    #[test]
    fn argmin() {
        let delta_tree_length =
            dvector![-50.0, -38.0, -38.0, -34.0, -34.0, -40.0, -34.0, -34.0, -40.0, -48.0];
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::argmin(delta_tree_length),
            0
        );
        let same_tree_length =
            dvector![-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0];
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::argmin(same_tree_length),
            0
        );
        let weird_tree_length = dvector![10.0, 15.0, 3.0, 20.0, 40.0, 500.0, 1000.0, 30.0];
        assert_eq!(
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::argmin(weird_tree_length),
            2
        )
    }

    #[test]
    fn softmax() {
        let delta_tree_length = dvector![1.3, 5.1, 2.2, 0.7, 1.1];
        let softmax_vector =
            NJTreeBuilder::<LevenshteinDNACorrected, DefaultGenerator>::softmax(delta_tree_length);
        assert_eq!(
            softmax_vector,
            dvector![
                0.020190464732580685,
                0.9025376890165726,
                0.04966052987196013,
                0.011080761983386346,
                0.01653055439550022
            ]
        );
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
        let nj_distances = DistanceMatrix {
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
        let rng = DefaultGenerator::new(3);
        let nj_builder = NJTreeBuilder::new_with_softmax(LevenshteinDNACorrected, &rng, 0.0);
        let nj_tree = nj_builder
            .build_from_distances(nj_distances, &sequences)
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
