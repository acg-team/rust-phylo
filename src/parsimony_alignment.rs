macro_rules! align {
    (@collect -) => { None };
    (@collect $l:tt) => { Some($l) };
    ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
}

mod alignment;
mod parsimony_info;

use bio::io::fasta::Record;
use rand::prelude::*;

use crate::{
    sequences::{get_parsimony_sets, SequenceType},
    tree::{self, Tree},
};

use alignment::{Alignment};
use parsimony_info::{ParsAlignSiteInfo, ParsimonyAlignmentMatrices};

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
    mismatch_cost: f32,
    gap_open_cost: f32,
    gap_ext_cost: f32,
    left_child_info: &[ParsAlignSiteInfo],
    right_child_info: &[ParsAlignSiteInfo],
    rng: fn(usize) -> usize,
) -> (Vec<ParsAlignSiteInfo>, Alignment, f32) {
    let mut pars_mats = ParsimonyAlignmentMatrices::new(
        left_child_info.len() + 1,
        right_child_info.len() + 1,
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        rng,
    );
    pars_mats.fill_matrices(left_child_info, right_child_info);
    pars_mats.traceback(left_child_info, right_child_info)
}

fn pars_align(
    mismatch_cost: f32,
    gap_open_cost: f32,
    gap_ext_cost: f32,
    left_child_info: &[ParsAlignSiteInfo],
    right_child_info: &[ParsAlignSiteInfo],
) -> (Vec<ParsAlignSiteInfo>, Alignment, f32) {
    pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        left_child_info,
        right_child_info,
        rng_len,
    )
}

pub(crate) fn pars_align_on_tree(
    mismatch_cost: f32,
    gap_open_cost: f32,
    gap_ext_cost: f32,
    tree: &tree::Tree,
    sequences: &[Record],
    sequence_type: &SequenceType,
) -> (Vec<Alignment>, Vec<f32>) {
    let num = tree.nodes.len();
    let order = &tree.postorder;

    assert_eq!(num, order.len());

    let mut node_info = vec![Vec::<ParsAlignSiteInfo>::new(); num];
    let mut alignments = vec![Alignment::empty(); num];
    let mut scores = vec![0.0; num];

    for &node_idx in order {
        if tree.is_leaf(node_idx) {
            let pars_sets = get_parsimony_sets(&sequences[node_idx], sequence_type);
            node_info[node_idx] = pars_sets
                .into_iter()
                .map(ParsAlignSiteInfo::new_leaf)
                .collect();
        } else {
            let ch1_idx = tree.nodes[node_idx].children[0];
            let ch2_idx = tree.nodes[node_idx].children[1];
            let (info, alignment, score) = pars_align(
                mismatch_cost,
                gap_open_cost,
                gap_ext_cost,
                &node_info[ch1_idx],
                &node_info[ch2_idx],
            );
            node_info[node_idx] = info;
            alignments[node_idx] = alignment;
            scores[node_idx] = score;
        }
    }
    (alignments, scores)
}

pub(crate) fn compile_alignment(
    tree: &tree::Tree,
    sequences: &[Record],
    alignment: &[Alignment],
    subroot: Option<usize>,
) -> Vec<Record> {
    let subroot_idx = match subroot {
        Some(idx) => idx,
        None => tree.root,
    };
    let order = tree.preorder_subroot(subroot_idx);
    let mut alignment_stack = vec![Vec::<Option<usize>>::new(); tree.nodes.len()];
    alignment_stack[subroot_idx] = (0..alignment[subroot_idx].map_x.len()).map(Some).collect();
    let mut msa = Vec::<Record>::with_capacity(tree.leaf_number);
    for &node_idx in order {
        if tree.is_leaf(node_idx) {
            let mut sequence = vec![b'-'; alignment_stack[subroot_idx].len()];
            for (alignment_index, site) in alignment_stack[node_idx].iter().enumerate() {
                if let Some(index) = site {
                    sequence[alignment_index] = sequences[node_idx].seq()[*index]
                }
            }
            msa.push(Record::with_attrs(
                sequences[node_idx].id(),
                sequences[node_idx].desc(),
                &sequence,
            ));
        } else {
            let mut padded_map_x = vec![None; alignment_stack[node_idx].len()];
            let mut padded_map_y = vec![None; alignment_stack[node_idx].len()];
            for (mapping_index, site) in alignment_stack[node_idx].iter().enumerate() {
                if let Some(index) = site {
                    padded_map_x[mapping_index] = alignment[node_idx].map_x[*index];
                    padded_map_y[mapping_index] = alignment[node_idx].map_y[*index];
                }
            }
            alignment_stack[tree.nodes[node_idx].children[0]] = padded_map_x;
            alignment_stack[tree.nodes[node_idx].children[1]] = padded_map_y;
        }
    }
    msa
}

#[allow(dead_code)]
fn setup_test_tree() -> (Tree, Vec<Record>, Vec<Alignment>) {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ];
    let mut tree = Tree::new(5, 8);
    tree.add_parent(5, 0, 1, 1.0, 1.0);
    tree.add_parent(6, 3, 4, 1.0, 1.0);
    tree.add_parent(7, 2, 6, 1.0, 1.0);
    tree.add_parent(8, 5, 7, 1.0, 1.0);
    tree.create_postorder();
    tree.create_preorder();
    // ((0:1.0, 1:1.0)5:1.0,(2:1.0,(3:1.0, 4:1.0)6:1.0)7:1.0)8:1.0;
    let alignment = vec![
        Alignment::empty(),
        Alignment::empty(),
        Alignment::empty(),
        Alignment::empty(),
        Alignment::empty(),
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
fn alignment_compile_root() {
    let (tree, sequences, alignment) = setup_test_tree();
    let msa = compile_alignment(&tree, &sequences, &alignment, None);
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
    assert_eq!(msa[2].seq(), "AA---".as_bytes());
    assert_eq!(msa[3].seq(), "---A-".as_bytes());
    assert_eq!(msa[4].seq(), "-A-AA".as_bytes());
}

#[test]
fn alignment_compile_internal1() {
    let (tree, sequences, alignment) = setup_test_tree();
    let msa = compile_alignment(&tree, &sequences, &alignment, Some(5));
    for seq in &msa {
        println!("{}", seq);
    }
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
}

#[test]
fn alignment_compile_internal2() {
    let (tree, sequences, alignment) = setup_test_tree();
    let msa = compile_alignment(&tree, &sequences, &alignment, Some(6));
    for seq in &msa {
        println!("{}", seq);
    }
    assert_eq!(msa[0].seq(), "-A-".as_bytes());
    assert_eq!(msa[1].seq(), "AAA".as_bytes());
}

#[test]
fn align_two_first_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;
    let sequences = [
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("B", None, b"AC"),
    ];
    let leaf_info1: Vec<ParsAlignSiteInfo> = get_parsimony_sets(&sequences[0], &SequenceType::DNA)
        .into_iter()
        .map(ParsAlignSiteInfo::new_leaf)
        .collect();
    let leaf_info2: Vec<ParsAlignSiteInfo> = get_parsimony_sets(&sequences[1], &SequenceType::DNA)
        .into_iter()
        .map(ParsAlignSiteInfo::new_leaf)
        .collect();
    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |l| l - 1,
    );
    assert_eq!(score, 3.5);
    assert_eq!(alignment.map_x.len(), 4);
    assert_eq!(alignment.map_y.len(), 4);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - -));
}

#[test]
fn align_two_second_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;
    let sequences = [
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("B", None, b"AC"),
    ];
    let leaf_info1: Vec<ParsAlignSiteInfo> = get_parsimony_sets(&sequences[0], &SequenceType::DNA)
        .into_iter()
        .map(ParsAlignSiteInfo::new_leaf)
        .collect();
    let leaf_info2: Vec<ParsAlignSiteInfo> = get_parsimony_sets(&sequences[1], &SequenceType::DNA)
        .into_iter()
        .map(ParsAlignSiteInfo::new_leaf)
        .collect();
    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |_| 0,
    );
    assert_eq!(score, 3.5);
    assert_eq!(alignment.map_x.len(), 4);
    assert_eq!(alignment.map_y.len(), 4);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 - -1));
}

#[test]
fn align_two_on_tree() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let sequences = [
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("A", None, b"AC"),
    ];
    let mut tree = Tree::new(2, 2);
    tree.add_parent(2, 0, 1, 1.0, 1.0);
    tree.create_postorder();
    let (alignment_vec, score) = pars_align_on_tree(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &tree,
        &sequences,
        &SequenceType::DNA,
    );
    assert_eq!(score[tree.root], 3.5);
    let alignment = &alignment_vec[tree.root];
    assert_eq!(alignment.map_x.len(), 4);
    assert_eq!(alignment.map_y.len(), 4);
}

#[test]
fn internal_alignment_first_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let leaf_info1 = create_site_info(&[
        (4u8, false, false),
        (6u8, false, false),
        (2u8, true, false),
        (1u8, true, false),
    ]);

    let leaf_info2 = create_site_info(&[(8u8, true, false), (4u8, false, false)]);

    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |_| 0,
    );
    assert_eq!(score, 1.0);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - -));
}

#[test]
fn internal_alignment_second_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let leaf_info1 = create_site_info(&[
        (4u8, false, false),
        (4u8, true, false),
        (2u8, true, false),
        (3u8, false, false),
    ]);

    let leaf_info2 = create_site_info(&[(8u8, true, false), (4u8, false, false)]);

    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |_| 0,
    );
    assert_eq!(score, 2.0);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 - -1));
}

#[test]
fn internal_alignment_third_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let leaf_info1 = create_site_info(&[
        (4u8, false, false),
        (4u8, true, false),
        (2u8, true, false),
        (3u8, false, false),
    ]);

    let leaf_info2 = create_site_info(&[(8u8, true, false), (4u8, false, false)]);

    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |l| l - 1,
    );
    assert_eq!(score, 2.0);
    assert_eq!(alignment.map_x, align!(- 0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - - -));
}

#[allow(dead_code)]
fn create_site_info(info: &[(u8, bool, bool)]) -> Vec<ParsAlignSiteInfo> {
    info.into_iter()
        .map(|(set, poss, perm)| ParsAlignSiteInfo::new(*set, *poss, *perm))
        .collect()
}

#[test]
fn align_four_on_tree() {
    let a = 2.0;
    let b = 0.5;
    let c = 1.0;

    let sequences = [
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("B", None, b"AC"),
        Record::with_attrs("C", None, b"A"),
        Record::with_attrs("D", None, b"GA"),
    ];

    let mut tree = Tree::new(4, 6);
    tree.add_parent(4, 0, 1, 1.0, 1.0);
    tree.add_parent(5, 2, 3, 1.0, 1.0);
    tree.add_parent(6, 4, 5, 1.0, 1.0);
    tree.create_postorder();

    let (alignment_vec, score) = pars_align_on_tree(c, a, b, &tree, &sequences, &SequenceType::DNA);
    // first cherry
    assert_eq!(score[4], 3.5);
    assert_eq!(alignment_vec[4].map_x.len(), 4);
    // second cherry
    assert_eq!(score[5], 2.0);
    assert_eq!(alignment_vec[5].map_x.len(), 2);
    // root, three possible alignments
    assert!(score[6] == 1.0 || score[6] == 2.0);
    if score[6] == 1.0 {
        assert_eq!(alignment_vec[6].map_x.len(), 4);
    } else {
        assert!(alignment_vec[6].map_x.len() == 4 || alignment_vec[6].map_x.len() == 5);
    }
}
