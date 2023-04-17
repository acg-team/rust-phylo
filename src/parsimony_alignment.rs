macro_rules! align {
    (@collect -) => { None };
    (@collect $l:tt) => { Some($l) };
    ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
}

mod alignment;
mod parsimony_info;

use bio::io::fasta::Record;
use rand::prelude::*;

use tree::NodeIdx::Internal as Int;
use tree::NodeIdx::Leaf;

use crate::{
    sequences::{get_parsimony_sets, SequenceType},
    tree::{self, NodeIdx, Tree},
};

use alignment::Alignment;
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
    let order = &tree.postorder;

    assert_eq!(tree.internals.len() + tree.leaves.len(), order.len());

    let mut internal_info = vec![Vec::<ParsAlignSiteInfo>::new(); tree.internals.len()];
    let mut leaf_info = vec![Vec::<ParsAlignSiteInfo>::new(); tree.leaves.len()];
    let mut alignments = vec![Alignment::empty(); tree.internals.len()];
    let mut scores = vec![0.0; tree.internals.len()];

    for &node_idx in order {
        match node_idx {
            Int(idx) => {
                let ch1_idx = tree.internals[idx].children[0];
                let ch2_idx = tree.internals[idx].children[1];
                let (info, alignment, score) = pars_align(
                    mismatch_cost,
                    gap_open_cost,
                    gap_ext_cost,
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
                    .map(ParsAlignSiteInfo::new_leaf)
                    .collect();
            }
        }
    }
    (alignments, scores)
}

pub(crate) fn compile_alignment(
    tree: &tree::Tree,
    sequences: &[Record],
    alignment: &[Alignment],
    subroot: Option<NodeIdx>,
) -> Vec<Record> {
    let subroot_idx = match subroot {
        Some(idx) => idx,
        None => tree.root,
    };
    let order = tree.preorder_subroot(subroot_idx);
    let mut alignment_stack =
        vec![Vec::<Option<usize>>::new(); tree.internals.len() + tree.leaves.len()];

    match subroot_idx {
        Int(idx) => alignment_stack[idx] = (0..alignment[idx].map_x.len()).map(Some).collect(),
        Leaf(idx) => return vec![sequences[idx].clone()],
    }

    let mut msa = Vec::<Record>::with_capacity(tree.leaf_number);
    for node_idx in order {
        match node_idx {
            Int(idx) => {
                let mut padded_map_x = vec![None; alignment_stack[idx].len()];
                let mut padded_map_y = vec![None; alignment_stack[idx].len()];
                for (mapping_index, site) in alignment_stack[idx].iter().enumerate() {
                    if let Some(index) = site {
                        padded_map_x[mapping_index] = alignment[idx].map_x[*index];
                        padded_map_y[mapping_index] = alignment[idx].map_y[*index];
                    }
                }
                match tree.internals[idx].children[0] {
                    Int(child_idx) => alignment_stack[child_idx] = padded_map_x,
                    Leaf(child_idx) => {
                        alignment_stack[tree.internals.len() + child_idx] = padded_map_x
                    }
                }
                match tree.internals[idx].children[1] {
                    Int(child_idx) => alignment_stack[child_idx] = padded_map_y,
                    Leaf(child_idx) => {
                        alignment_stack[tree.internals.len() + child_idx] = padded_map_y
                    }
                }
            }
            Leaf(idx) => {
                let mut sequence = vec![b'-'; alignment_stack[tree.internals.len() + idx].len()];
                for (alignment_index, site) in alignment_stack[tree.internals.len() + idx]
                    .iter()
                    .enumerate()
                {
                    if let Some(index) = site {
                        sequence[alignment_index] = sequences[idx].seq()[*index]
                    }
                }
                msa.push(Record::with_attrs(
                    sequences[idx].id(),
                    sequences[idx].desc(),
                    &sequence,
                ));
            }
        }
    }
    msa.sort_by(|a, b| sequence_idx(sequences, a).cmp(&sequence_idx(sequences, b)));
    msa
}

fn sequence_idx(sequences: &[Record], search: &Record) -> usize {
    sequences
        .iter()
        .position(|r| r.id() == search.id())
        .unwrap()
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
    let mut tree = Tree::new(5, 3);
    tree.add_parent(0, Leaf(0), Leaf(1), 1.0, 1.0);
    tree.add_parent(1, Leaf(3), Leaf(4), 1.0, 1.0);
    tree.add_parent(2, Leaf(2), Int(1), 1.0, 1.0);
    tree.add_parent(3, Int(0), Int(2), 1.0, 1.0);
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
fn alignment_compile_root() {
    let (tree, sequences, alignment) = setup_test_tree();
    let msa = compile_alignment(&tree, &sequences, &alignment, None);
    println!("{:?}", msa);
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
    assert_eq!(msa[2].seq(), "AA---".as_bytes());
    assert_eq!(msa[3].seq(), "---A-".as_bytes());
    assert_eq!(msa[4].seq(), "-A-AA".as_bytes());
}

#[test]
fn alignment_compile_internal1() {
    let (tree, sequences, alignment) = setup_test_tree();
    let msa = compile_alignment(&tree, &sequences, &alignment, Some(Int(0)));
    for seq in &msa {
        println!("{}", seq);
    }
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
}

#[test]
fn alignment_compile_internal2() {
    let (tree, sequences, alignment) = setup_test_tree();
    let msa = compile_alignment(&tree, &sequences, &alignment, Some(Int(1)));
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
    let mut tree = Tree::new(2, 0);
    tree.add_parent(0, Leaf(0), Leaf(1), 1.0, 1.0);
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

#[allow(dead_code)]
fn create_site_info(info: &[(u8, bool, bool)]) -> Vec<ParsAlignSiteInfo> {
    info.into_iter()
        .map(|(set, poss, perm)| ParsAlignSiteInfo::new(*set, *poss, *perm))
        .collect()
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

    let mut tree = Tree::new(4, 2);
    tree.add_parent(0, Leaf(0), Leaf(1), 1.0, 1.0);
    tree.add_parent(1, Leaf(2), Leaf(3), 1.0, 1.0);
    tree.add_parent(2, Int(0), Int(1), 1.0, 1.0);
    tree.create_postorder();

    let (alignment_vec, score) = pars_align_on_tree(c, a, b, &tree, &sequences, &SequenceType::DNA);
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
