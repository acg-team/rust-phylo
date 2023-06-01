pub(crate) mod parsimony_info;
pub(crate) mod parsimony_matrices;
pub(crate) mod parsimony_sets;

use bio::io::fasta::Record;
// use nalgebra::SMatrix;
use rand::prelude::*;

use crate::{
    parsimony_alignment::parsimony_sets::get_parsimony_sets,
    sequences::SequenceType,
    // substitution_models::{dna_models, protein_models},
    tree::{self, NodeIdx::Internal as Int, NodeIdx::Leaf},
};

use crate::alignment::Alignment;
use parsimony_info::ParsimonySiteInfo;
use parsimony_matrices::ParsimonyAlignmentMatrices;

// type CostMatrix<const N: usize> = SMatrix<f64, N, N>;

pub(crate) struct ParsimonyCostsSimple {
    mismatch: f64,
    gap_open: f64,
    gap_ext: f64,
}

impl ParsimonyCostsSimple {
    pub(crate) fn new_default() -> ParsimonyCostsSimple {
        Self::new(1.0, 2.5, 0.5)
    }

    pub(crate) fn new(mismatch: f64, gap_open: f64, gap_ext: f64) -> ParsimonyCostsSimple {
        ParsimonyCostsSimple {
            mismatch,
            gap_open,
            gap_ext,
        }
    }
}

// impl ParsimonyCostsSimple<20> {
//     fn new() -> ParsimonyCostsSimple<20> {
//         ParsimonyCostsSimple {
//             index: protein_models::aminoacid_index(),
//             c: ParsimonyCostsSimple::<20>::make_c(),
//             gap_open_mult: 0.5,
//             gap_ext_mult: 2.5,
//         }
//     }
// }

// impl ParsimonyCostsSimple<4> {
//     fn new() -> ParsimonyCostsSimple<4> {
//         ParsimonyCostsSimple {
//             index: dna_models::nucleotide_index(),
//             c: ParsimonyCostsSimple::<4>::make_c(),
//             gap_open_mult: 0.5,
//             gap_ext_mult: 2.5,
//         }
//     }
// }

// impl<const N: usize> ParsimonyCostsSimple<N> {
//     fn make_c() -> CostMatrix<N> {
//         let mut c = CostMatrix::<N>::from_element(1.0);
//         c.fill_diagonal(0.0);
//         c
//     }
// }

pub trait ParsimonyCosts {
    fn match_cost(&self, branch_length: f64, i: u8, j: u8) -> f64;
    fn gap_ext_cost(&self, branch_length: f64) -> f64;
    fn gap_open_cost(&self, branch_length: f64) -> f64;
}

impl ParsimonyCosts for ParsimonyCostsSimple {
    fn match_cost(&self, _: f64, i: u8, j: u8) -> f64 {
        if i == j {
            0.0
        } else {
            self.mismatch
        }
    }
    fn gap_ext_cost(&self, _: f64) -> f64 {
        self.gap_ext
    }

    fn gap_open_cost(&self, _: f64) -> f64 {
        self.gap_open
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapX,
    GapY,
    Skip,
}

fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}

fn pars_align_w_rng(
    scoring: &impl ParsimonyCosts,
    left_child_info: &[ParsimonySiteInfo],
    right_child_info: &[ParsimonySiteInfo],
    rng: fn(usize) -> usize,
) -> (Vec<ParsimonySiteInfo>, Alignment, f64) {
    let mut pars_mats =
        ParsimonyAlignmentMatrices::new(left_child_info.len() + 1, right_child_info.len() + 1, rng);
    pars_mats.fill_matrices(left_child_info, right_child_info, scoring);
    pars_mats.traceback(left_child_info, right_child_info)
}

fn pars_align(
    scoring: &impl ParsimonyCosts,
    left_child_info: &[ParsimonySiteInfo],
    right_child_info: &[ParsimonySiteInfo],
) -> (Vec<ParsimonySiteInfo>, Alignment, f64) {
    pars_align_w_rng(scoring, left_child_info, right_child_info, rng_len)
}

pub(crate) fn pars_align_on_tree(
    mismatch_cost: f64,
    gap_open_cost: f64,
    gap_ext_cost: f64,
    tree: &tree::Tree,
    sequences: &[Record],
    sequence_type: &SequenceType,
) -> (Vec<Alignment>, Vec<f64>) {
    let order = &tree.postorder;
    let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

    assert_eq!(tree.internals.len() + tree.leaves.len(), order.len());

    let mut internal_info = vec![Vec::<ParsimonySiteInfo>::new(); tree.internals.len()];
    let mut leaf_info = vec![Vec::<ParsimonySiteInfo>::new(); tree.leaves.len()];
    let mut alignments = vec![Alignment::empty(); tree.internals.len()];
    let mut scores = vec![0.0; tree.internals.len()];

    for &node_idx in order {
        match node_idx {
            Int(idx) => {
                let ch1_idx = tree.internals[idx].children[0];
                let ch2_idx = tree.internals[idx].children[1];
                let (info, alignment, score) = pars_align(
                    &scoring,
                    match ch1_idx {
                        Int(idx1) => &internal_info[idx1],
                        Leaf(idx1) => &leaf_info[idx1],
                    },
                    match ch2_idx {
                        Int(idx2) => &internal_info[idx2],
                        Leaf(idx2) => &leaf_info[idx2],
                    },
                );

                internal_info[idx] = info;
                alignments[idx] = alignment;
                scores[idx] = score;
            }
            Leaf(idx) => {
                let pars_sets = get_parsimony_sets(&sequences[idx], sequence_type);
                leaf_info[idx] = pars_sets
                    .into_iter()
                    .map(ParsimonySiteInfo::new_leaf)
                    .collect();
            }
        }
    }
    (alignments, scores)
}

pub(crate) fn sequence_idx(sequences: &[Record], search: &Record) -> usize {
    sequences
        .iter()
        .position(|r| r.id() == search.id())
        .unwrap()
}

#[cfg(test)]
mod parsimony_alignment_tests {
    use crate::alignment::{compile_alignment_representation, Alignment};
    use crate::parsimony_alignment::{
        pars_align_on_tree, pars_align_w_rng, parsimony_info::ParsimonySiteInfo,
        parsimony_sets::get_parsimony_sets, ParsimonyCostsSimple,
    };
    use crate::sequences::SequenceType;
    use crate::tree::{NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree};

    use bio::io::fasta::Record;

    macro_rules! align {
        (@collect -) => { None };
        (@collect $l:tt) => { Some($l) };
        ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
    }

    pub(crate) fn setup_test_tree() -> (Tree, Vec<Record>, Vec<Alignment>) {
        let sequences = vec![
            Record::with_attrs("A0", None, b"AAAAA"),
            Record::with_attrs("B1", None, b"A"),
            Record::with_attrs("C2", None, b"AA"),
            Record::with_attrs("D3", None, b"A"),
            Record::with_attrs("E4", None, b"AAA"),
        ];
        let mut tree = Tree::new(5, 3);
        tree.add_parent(0, L(0), L(1), 1.0, 1.0);
        tree.add_parent(1, L(3), L(4), 1.0, 1.0);
        tree.add_parent(2, L(2), I(1), 1.0, 1.0);
        tree.add_parent(3, I(0), I(2), 1.0, 1.0);
        tree.create_postorder();
        tree.create_preorder();
        // ((0:1.0, 1:1.0)5:1.0,(2:1.0,(3:1.0, 4:1.0)6:1.0)7:1.0)8:1.0;
        let alignment = vec![
            Alignment::new(align!(0 1 2 3 4), align!(- - - 0 -)),
            Alignment::new(align!(- 0 -), align!(0 1 2)),
            Alignment::new(align!(0 1 - -), align!(- 0 1 2)),
            Alignment::new(align!(0 1 2 3 4), align!(0 1 - 2 3)),
        ];
        // A0> AAAAA
        // B1> ---A-
        // C2> AA---
        // D3> ---A-
        // E4> -A-AA
        (tree, sequences, alignment)
    }

    #[test]
    pub(crate) fn alignment_compile_root() {
        let (tree, sequences, alignment) = setup_test_tree();
        let msa = compile_alignment_representation(&tree, &sequences, &alignment, None);
        assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
        assert_eq!(msa[1].seq(), "---A-".as_bytes());
        assert_eq!(msa[2].seq(), "AA---".as_bytes());
        assert_eq!(msa[3].seq(), "---A-".as_bytes());
        assert_eq!(msa[4].seq(), "-A-AA".as_bytes());
    }

    #[test]
    pub(crate) fn alignment_compile_internal1() {
        let (tree, sequences, alignment) = setup_test_tree();
        let msa = compile_alignment_representation(&tree, &sequences, &alignment, Some(I(0)));
        assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
        assert_eq!(msa[1].seq(), "---A-".as_bytes());
    }

    #[test]
    pub(crate) fn alignment_compile_internal2() {
        let (tree, sequences, alignment) = setup_test_tree();
        let msa = compile_alignment_representation(&tree, &sequences, &alignment, Some(I(1)));
        assert_eq!(msa[0].seq(), "-A-".as_bytes());
        assert_eq!(msa[1].seq(), "AAA".as_bytes());
    }

    #[test]
    pub(crate) fn align_two_first_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let sequences = [
            Record::with_attrs("A", None, b"AACT"),
            Record::with_attrs("B", None, b"AC"),
        ];
        let leaf_info1: Vec<ParsimonySiteInfo> =
            get_parsimony_sets(&sequences[0], &SequenceType::DNA)
                .into_iter()
                .map(ParsimonySiteInfo::new_leaf)
                .collect();
        let leaf_info2: Vec<ParsimonySiteInfo> =
            get_parsimony_sets(&sequences[1], &SequenceType::DNA)
                .into_iter()
                .map(ParsimonySiteInfo::new_leaf)
                .collect();
        let (_info, alignment, score) =
            pars_align_w_rng(&scoring, &leaf_info1, &leaf_info2, |l| l - 1);
        assert_eq!(score, 3.5);
        assert_eq!(alignment.map_x.len(), 4);
        assert_eq!(alignment.map_y.len(), 4);
        assert_eq!(alignment.map_x, align!(0 1 2 3));
        assert_eq!(alignment.map_y, align!(0 1 - -));
    }

    #[test]
    pub(crate) fn align_two_second_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);
        let sequences = [
            Record::with_attrs("A", None, b"AACT"),
            Record::with_attrs("B", None, b"AC"),
        ];
        let leaf_info1: Vec<ParsimonySiteInfo> =
            get_parsimony_sets(&sequences[0], &SequenceType::DNA)
                .into_iter()
                .map(ParsimonySiteInfo::new_leaf)
                .collect();
        let leaf_info2: Vec<ParsimonySiteInfo> =
            get_parsimony_sets(&sequences[1], &SequenceType::DNA)
                .into_iter()
                .map(ParsimonySiteInfo::new_leaf)
                .collect();
        let (_info, alignment, score) = pars_align_w_rng(&scoring, &leaf_info1, &leaf_info2, |_| 0);
        assert_eq!(score, 3.5);
        assert_eq!(alignment.map_x.len(), 4);
        assert_eq!(alignment.map_y.len(), 4);
        assert_eq!(alignment.map_x, align!(0 1 2 3));
        assert_eq!(alignment.map_y, align!(0 - -1));
    }

    #[test]
    pub(crate) fn align_two_on_tree() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;

        let sequences = [
            Record::with_attrs("A", None, b"AACT"),
            Record::with_attrs("A", None, b"AC"),
        ];
        let mut tree = Tree::new(2, 0);
        tree.add_parent(0, L(0), L(1), 1.0, 1.0);
        tree.create_postorder();
        let (alignment_vec, score) = pars_align_on_tree(
            mismatch_cost,
            gap_open_cost,
            gap_ext_cost,
            &tree,
            &sequences,
            &SequenceType::DNA,
        );
        assert_eq!(score[Into::<usize>::into(tree.root)], 3.5);
        let alignment = &alignment_vec[Into::<usize>::into(tree.root)];
        assert_eq!(alignment.map_x.len(), 4);
        assert_eq!(alignment.map_y.len(), 4);
    }

    #[test]
    pub(crate) fn internal_alignment_first_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let leaf_info1 = create_site_info(&[
            (8, false, false),
            (12, false, false),
            (4, true, false),
            (2, true, false),
        ]);

        let leaf_info2 = create_site_info(&[(16, true, false), (8, false, false)]);

        let (_info, alignment, score) = pars_align_w_rng(&scoring, &leaf_info1, &leaf_info2, |_| 0);
        assert_eq!(score, 1.0);
        assert_eq!(alignment.map_x, align!(0 1 2 3));
        assert_eq!(alignment.map_y, align!(0 1 - -));
    }

    #[allow(dead_code)]
    pub(crate) fn create_site_info(info: &[(u32, bool, bool)]) -> Vec<ParsimonySiteInfo> {
        info.into_iter()
            .map(|(set, poss, perm)| ParsimonySiteInfo::new(*set, *poss, *perm))
            .collect()
    }

    #[test]
    pub(crate) fn internal_alignment_second_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let leaf_info1 = create_site_info(&[
            (8, false, false),
            (8, true, false),
            (4, true, false),
            (6, false, false),
        ]);

        let leaf_info2 = create_site_info(&[(16, true, false), (8, false, false)]);

        let (_info, alignment, score) = pars_align_w_rng(&scoring, &leaf_info1, &leaf_info2, |_| 0);
        assert_eq!(score, 2.0);
        assert_eq!(alignment.map_x, align!(0 1 2 3));
        assert_eq!(alignment.map_y, align!(0 - -1));
    }

    #[test]
    pub(crate) fn internal_alignment_third_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let leaf_info1 = create_site_info(&[
            (8, false, false),
            (8, true, false),
            (4, true, false),
            (6, false, false),
        ]);

        let leaf_info2 = create_site_info(&[(16, true, false), (8, false, false)]);

        let (_info, alignment, score) =
            pars_align_w_rng(&scoring, &leaf_info1, &leaf_info2, |l| l - 1);
        assert_eq!(score, 2.0);
        assert_eq!(alignment.map_x, align!(- 0 1 2 3));
        assert_eq!(alignment.map_y, align!(0 1 - - -));
    }

    #[test]
    pub(crate) fn align_four_on_tree() {
        let a = 2.0;
        let b = 0.5;
        let c = 1.0;

        let sequences = [
            Record::with_attrs("A", None, b"AACT"),
            Record::with_attrs("B", None, b"AC"),
            Record::with_attrs("C", None, b"A"),
            Record::with_attrs("D", None, b"GA"),
        ];

        let mut tree = Tree::new(4, 2);
        tree.add_parent(0, L(0), L(1), 1.0, 1.0);
        tree.add_parent(1, L(2), L(3), 1.0, 1.0);
        tree.add_parent(2, I(0), I(1), 1.0, 1.0);
        tree.create_postorder();

        let (alignment_vec, score) =
            pars_align_on_tree(c, a, b, &tree, &sequences, &SequenceType::DNA);
        // first cherry
        assert_eq!(score[0], 3.5);
        assert_eq!(alignment_vec[0].map_x.len(), 4);
        // second cherry
        assert_eq!(score[1], 2.0);
        assert_eq!(alignment_vec[1].map_x.len(), 2);
        // root, three possible alignments
        assert!(score[2] == 1.0 || score[2] == 2.0);
        if score[2] == 1.0 {
            assert_eq!(alignment_vec[2].map_x.len(), 4);
        } else {
            assert!(alignment_vec[2].map_x.len() == 4 || alignment_vec[2].map_x.len() == 5);
        }
    }
}
