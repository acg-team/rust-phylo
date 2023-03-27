use bio::io::fasta::Record;
use rand::prelude::*;

use crate::{
    sequences::{self, SequenceType},
    tree,
};

mod parsimony_info;

use parsimony_info::ParsAlignSiteInfo;
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapX,
    GapY,
    Skip,
}

pub(crate) type Alignment = Vec<(Option<usize>, Option<usize>)>;

fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}

fn pars_align_w_rng(
    mismatch_cost: f32,
    gap_open_cost: f32,
    gap_ext_cost: f32,
    left_child_info: &[parsimony_info::ParsAlignSiteInfo],
    right_child_info: &[parsimony_info::ParsAlignSiteInfo],
    rng: fn(usize) -> usize,
) -> (Vec<parsimony_info::ParsAlignSiteInfo>, Alignment, f32) {
    let mut pars_mats = parsimony_info::ParsimonyAlignmentMatrices::new(
        left_child_info.len() + 1,
        right_child_info.len() + 1,
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        rng,
    );
    pars_mats.fill_matrices(left_child_info, right_child_info);
    println!("{}", pars_mats);
    pars_mats.traceback(left_child_info, right_child_info)
}

fn pars_align(
    mismatch_cost: f32,
    gap_open_cost: f32,
    gap_ext_cost: f32,
    left_child_info: &[parsimony_info::ParsAlignSiteInfo],
    right_child_info: &[parsimony_info::ParsAlignSiteInfo],
) -> (Vec<parsimony_info::ParsAlignSiteInfo>, Alignment, f32) {
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
    let mut alignments = vec![Alignment::new(); num];
    let mut scores = vec![0.0; num];

    for &node_idx in order {
        println!("{}", node_idx);
        if tree.is_leaf(node_idx) {
            let pars_sets = sequences::get_parsimony_sets(&sequences[node_idx], sequence_type);
            node_info[node_idx] = pars_sets
                .into_iter()
                .map(|set| parsimony_info::ParsAlignSiteInfo::new_leaf(set))
                .collect();
        } else {
            let ch1_idx = tree.nodes[node_idx].children[0];
            let ch2_idx = tree.nodes[node_idx].children[1];

            println!("{:?}", node_info[ch1_idx]);
            println!("{:?}", node_info[ch2_idx]);
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
            println!("{:?}", node_info[node_idx]);
            println!("Score for this alignment is {}", score);

            if tree.is_leaf(tree.nodes[node_idx].children[0])
                && tree.is_leaf(tree.nodes[node_idx].children[1])
            {
                print_alignment(
                    sequences[tree.nodes[node_idx].children[0]].seq(),
                    sequences[tree.nodes[node_idx].children[1]].seq(),
                    &alignments[node_idx],
                );
            }
        }
    }
    (alignments, scores)
}

fn print_alignment(sequence1: &[u8], sequence2: &[u8], alignment: &Alignment) {
    let mut one = Vec::<char>::with_capacity(alignment.len());
    let mut two = Vec::<char>::with_capacity(alignment.len());
    for (site1, site2) in alignment {
        match site1 {
            Some(index) => one.push(sequence1[*index] as char),
            None => one.push('-'),
        };
        match site2 {
            Some(index) => two.push(sequence2[*index] as char),
            None => two.push('-'),
        };
    }
    println!("{}", String::from_iter(one));
    println!("{}", String::from_iter(two));
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
    let leaf_info1: Vec<parsimony_info::ParsAlignSiteInfo> =
        sequences::get_parsimony_sets(&sequences[0], &SequenceType::DNA)
            .into_iter()
            .map(|set| parsimony_info::ParsAlignSiteInfo::new_leaf(set))
            .collect();
    let leaf_info2: Vec<parsimony_info::ParsAlignSiteInfo> =
        sequences::get_parsimony_sets(&sequences[1], &SequenceType::DNA)
            .into_iter()
            .map(|set| parsimony_info::ParsAlignSiteInfo::new_leaf(set))
            .collect();
    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |l| l - 1,
    );
    print_alignment(&sequences[0].seq(), &sequences[1].seq(), &alignment);
    assert_eq!(score, 3.5);
    assert_eq!(alignment.len(), 4);
    assert_eq!(
        alignment,
        vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), None),
            (Some(3), None)
        ]
    );
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
    let leaf_info1: Vec<parsimony_info::ParsAlignSiteInfo> =
        sequences::get_parsimony_sets(&sequences[0], &SequenceType::DNA)
            .into_iter()
            .map(|set| parsimony_info::ParsAlignSiteInfo::new_leaf(set))
            .collect();
    let leaf_info2: Vec<parsimony_info::ParsAlignSiteInfo> =
        sequences::get_parsimony_sets(&sequences[1], &SequenceType::DNA)
            .into_iter()
            .map(|set| parsimony_info::ParsAlignSiteInfo::new_leaf(set))
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
    assert_eq!(alignment.len(), 4);
    assert_eq!(
        alignment,
        vec![
            (Some(0), Some(0)),
            (Some(1), None),
            (Some(2), None),
            (Some(3), Some(1))
        ]
    );
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

    let mut tree = tree::Tree::new(2, 2);
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
    assert_eq!(alignment.len(), 4);
}

#[test]
fn internal_alignment_first_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let leaf_info1 = vec![
        ParsAlignSiteInfo::new(4, false, false),
        ParsAlignSiteInfo::new(6, false, false),
        ParsAlignSiteInfo::new(2, true, false),
        ParsAlignSiteInfo::new(1, true, false),
    ];

    let leaf_info2 = vec![
        ParsAlignSiteInfo::new(8, true, false),
        ParsAlignSiteInfo::new(4, false, false),
    ];

    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |_| 0,
    );
    assert_eq!(score, 1.0);
    assert_eq!(
        alignment,
        vec![
            (Some(0), Some(0)),
            (Some(1), Some(1)),
            (Some(2), None),
            (Some(3), None)
        ]
    );
}

#[test]
fn internal_alignment_second_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let leaf_info1 = vec![
        ParsAlignSiteInfo::new(4, false, false),
        ParsAlignSiteInfo::new(4, true, false),
        ParsAlignSiteInfo::new(2, true, false),
        ParsAlignSiteInfo::new(3, false, false),
    ];

    let leaf_info2 = vec![
        ParsAlignSiteInfo::new(8, true, false),
        ParsAlignSiteInfo::new(4, false, false),
    ];

    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |_| 0,
    );
    assert_eq!(score, 2.0);
    assert_eq!(
        alignment,
        vec![
            (Some(0), Some(0)),
            (Some(1), None),
            (Some(2), None),
            (Some(3), Some(1))
        ]
    );
}

#[test]
fn internal_alignment_third_outcome() {
    let mismatch_cost = 1.0;
    let gap_open_cost = 2.0;
    let gap_ext_cost = 0.5;

    let leaf_info1 = vec![
        ParsAlignSiteInfo::new(4, false, false),
        ParsAlignSiteInfo::new(4, true, false),
        ParsAlignSiteInfo::new(2, true, false),
        ParsAlignSiteInfo::new(3, false, false),
    ];

    let leaf_info2 = vec![
        ParsAlignSiteInfo::new(8, true, false),
        ParsAlignSiteInfo::new(4, false, false),
    ];

    let (_info, alignment, score) = pars_align_w_rng(
        mismatch_cost,
        gap_open_cost,
        gap_ext_cost,
        &leaf_info1,
        &leaf_info2,
        |l| l-1,
    );
    assert_eq!(score, 2.0);
    assert_eq!(
        alignment,
        vec![
            (None, Some(0)),
            (Some(0), Some(1)),
            (Some(1), None),
            (Some(2), None),
            (Some(3), None)
        ]
    );
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

    let mut tree = tree::Tree::new(4, 6);
    tree.add_parent(4, 0, 1, 1.0, 1.0);
    tree.add_parent(5, 2, 3, 1.0, 1.0);
    tree.add_parent(6, 4, 5, 1.0, 1.0);
    tree.create_postorder();

    let (alignment_vec, score) = pars_align_on_tree(c, a, b, &tree, &sequences, &SequenceType::DNA);
    // first cherry
    assert_eq!(score[4], 3.5);
    assert_eq!(alignment_vec[4].len(), 4);
    // second cherry
    assert_eq!(score[5], 2.0);
    assert_eq!(alignment_vec[5].len(), 2);
    // root, three possible alignments
    assert!(score[6] == 1.0 || score[6] == 2.0);
    if score[6] == 1.0 {
        assert_eq!(alignment_vec[6].len(), 4);
    } else {
        assert!(alignment_vec[6].len() == 4 || alignment_vec[6].len() == 5);
    }
}
