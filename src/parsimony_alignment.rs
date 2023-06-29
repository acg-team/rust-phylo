pub(crate) mod parsimony_costs;
pub(crate) mod parsimony_info;
pub(crate) mod parsimony_matrices;
pub(crate) mod parsimony_sets;

use bio::io::fasta::Record;
use rand::prelude::*;

use crate::{
    parsimony_alignment::parsimony_sets::get_parsimony_sets,
    sequences::SequenceType,
    tree::{self, NodeIdx::Internal as Int, NodeIdx::Leaf},
};

use crate::alignment::Alignment;
use parsimony_costs::ParsimonyCosts;
use parsimony_info::ParsimonySiteInfo;
use parsimony_matrices::ParsimonyAlignmentMatrices;

use parsimony_costs::BranchParsimonyCosts;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapX,
    GapY,
}

fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}

fn pars_align_w_rng(
    l_info: &[ParsimonySiteInfo],
    l_scoring: &Box<&dyn BranchParsimonyCosts>,
    r_info: &[ParsimonySiteInfo],
    r_scoring: &Box<&dyn BranchParsimonyCosts>,
    rng: fn(usize) -> usize,
) -> (Vec<ParsimonySiteInfo>, Alignment, f64) {
    let mut pars_mats = ParsimonyAlignmentMatrices::new(l_info.len() + 1, r_info.len() + 1, rng);
    pars_mats.fill_matrices(l_info, l_scoring, r_info, r_scoring);
    pars_mats.traceback(l_info, r_info)
}

fn pars_align(
    l_info: &[ParsimonySiteInfo],
    l_scoring: &Box<&dyn BranchParsimonyCosts>,
    r_info: &[ParsimonySiteInfo],
    r_scoring: &Box<&dyn BranchParsimonyCosts>,
) -> (Vec<ParsimonySiteInfo>, Alignment, f64) {
    pars_align_w_rng(l_info, l_scoring, r_info, r_scoring, rng_len)
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
    let scoring = parsimony_costs::parsimony_costs_simple::ParsimonyCostsSimple::new(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
    );

    assert_eq!(tree.internals.len() + tree.leaves.len(), order.len());

    let mut internal_info = vec![Vec::<ParsimonySiteInfo>::new(); tree.internals.len()];
    let mut leaf_info = vec![Vec::<ParsimonySiteInfo>::new(); tree.leaves.len()];
    let mut alignments = vec![Alignment::empty(); tree.internals.len()];
    let mut scores = vec![0.0; tree.internals.len()];

    for &node_idx in order {
        match node_idx {
            Int(idx) => {
                let (l_info, left_branch) = match tree.internals[idx].children[0] {
                    Int(idx) => (&internal_info[idx], tree.internals[idx].blen),
                    Leaf(idx) => (&leaf_info[idx], tree.leaves[idx].blen),
                };
                let (r_info, right_branch) = match tree.internals[idx].children[1] {
                    Int(idx) => (&internal_info[idx], tree.internals[idx].blen),
                    Leaf(idx) => (&leaf_info[idx], tree.leaves[idx].blen),
                };
                let (info, alignment, score) = pars_align(
                    &l_info,
                    &scoring.get_branch_costs(left_branch),
                    &r_info,
                    &scoring.get_branch_costs(right_branch),
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
    use crate::parsimony_alignment::parsimony_costs::{
        parsimony_costs_simple::ParsimonyCostsSimple, ParsimonyCosts,
    };
    use crate::parsimony_alignment::{
        pars_align_on_tree, pars_align_w_rng, parsimony_info::ParsimonySiteInfo,
        parsimony_sets::get_parsimony_sets,
    };
    use crate::sequences::SequenceType;
    use crate::tree::{NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree};

    use bio::io::fasta::Record;

    use super::parsimony_info::GapFlag::{self, GapOpen, NoGap};

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
        let (_info, alignment, score) = pars_align_w_rng(
            &leaf_info1,
            &scoring.get_branch_costs(1.0),
            &leaf_info2,
            &scoring.get_branch_costs(1.0),
            |l| l - 1,
        );
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
        let (_info, alignment, score) = pars_align_w_rng(
            &leaf_info1,
            &scoring.get_branch_costs(1.0),
            &leaf_info2,
            &scoring.get_branch_costs(1.0),
            |_| 0,
        );
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

        let leaf_info1 = [
            (vec![b'A'], NoGap),
            (vec![b'C', b'A'], NoGap),
            (vec![b'C'], GapOpen),
            (vec![b'T'], GapOpen),
        ]
        .map(create_site_info);

        let leaf_info2 = [([b'G'], GapOpen), ([b'A'], NoGap)].map(create_site_info);

        let (_info, alignment, score) = pars_align_w_rng(
            &leaf_info1,
            &scoring.get_branch_costs(1.0),
            &leaf_info2,
            &scoring.get_branch_costs(1.0),
            |_| 0,
        );
        assert_eq!(score, 1.0);
        assert_eq!(alignment.map_x, align!(0 1 2 3));
        assert_eq!(alignment.map_y, align!(0 1 - -));
    }

    #[allow(dead_code)]
    pub(crate) fn create_site_info(
        args: (impl IntoIterator<Item = u8>, GapFlag),
    ) -> ParsimonySiteInfo {
        ParsimonySiteInfo::new(args.0, args.1)
    }

    #[test]
    pub(crate) fn internal_alignment_second_outcome() {
        let mismatch_cost = 1.0;
        let gap_open_cost = 2.0;
        let gap_ext_cost = 0.5;
        let scoring = ParsimonyCostsSimple::new(mismatch_cost, gap_open_cost, gap_ext_cost);

        let leaf_info1 = [
            (vec![b'A'], NoGap),
            (vec![b'A'], GapOpen),
            (vec![b'C'], GapOpen),
            (vec![b'T', b'C'], NoGap),
        ]
        .map(create_site_info);

        let leaf_info2 = [([b'G'], GapOpen), ([b'A'], NoGap)].map(create_site_info);

        let (_info, alignment, score) = pars_align_w_rng(
            &leaf_info1,
            &scoring.get_branch_costs(1.0),
            &leaf_info2,
            &scoring.get_branch_costs(1.0),
            |_| 0,
        );
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

        let leaf_info1 = [
            (vec![b'A'], NoGap),
            (vec![b'A'], GapOpen),
            (vec![b'C'], GapOpen),
            (vec![b'C', b'T'], NoGap),
        ]
        .map(create_site_info);

        let leaf_info2 = [(vec![b'G'], GapOpen), (vec![b'A'], NoGap)].map(create_site_info);

        let (_info, alignment, score) = pars_align_w_rng(
            &leaf_info1,
            &scoring.get_branch_costs(1.0),
            &leaf_info2,
            &scoring.get_branch_costs(1.0),
            |l| l - 1,
        );
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
