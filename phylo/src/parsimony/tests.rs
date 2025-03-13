use bio::io::fasta::Record;

use crate::alignment::Sequences;
use crate::alphabets::dna_alphabet as dna;
use crate::parsimony::costs::ParsimonyCostsSimple;
use crate::parsimony::SiteFlag::{self, GapOpen, NoGap};
use crate::parsimony::{pars_align_on_tree, pars_align_w_rng, ParsimonySite};
use crate::record_wo_desc as rec;
use crate::tree;
use crate::tree::tree_parser::from_newick;

macro_rules! align {
    (@collect -) => { None };
    (@collect $l:tt) => { Some($l) };
    ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
}

#[test]
pub(crate) fn align_two_first_outcome() {
    let mismatch = 1.0;
    let gap_open = 2.0;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let dna = dna();

    let sequences = [
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("B", None, b"AC"),
    ];

    let leaf_info1 = sequences[0]
        .seq()
        .iter()
        .map(|c| ParsimonySite::new_leaf(dna.parsimony_set(c)))
        .collect::<Vec<ParsimonySite>>();
    let leaf_info2 = sequences[1]
        .seq()
        .iter()
        .map(|c| ParsimonySite::new_leaf(dna.parsimony_set(c)))
        .collect::<Vec<ParsimonySite>>();
    let (_info, alignment, score) =
        pars_align_w_rng(&leaf_info1, 1.0, &leaf_info2, 1.0, &scoring, |l| l - 1);
    assert_eq!(score, 3.5);
    assert_eq!(alignment.map_x.len(), 4);
    assert_eq!(alignment.map_y.len(), 4);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - -));
}

#[test]
pub(crate) fn align_two_second_outcome() {
    let mismatch = 1.0;
    let gap_open = 2.0;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());
    let dna = dna();

    let sequences = [
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("B", None, b"AC"),
    ];

    let leaf_info1 = sequences[0]
        .seq()
        .iter()
        .map(|c| ParsimonySite::new_leaf(dna.parsimony_set(c)))
        .collect::<Vec<ParsimonySite>>();
    let leaf_info2 = sequences[1]
        .seq()
        .iter()
        .map(|c| ParsimonySite::new_leaf(dna.parsimony_set(c)))
        .collect::<Vec<ParsimonySite>>();
    let (_info, alignment, score) =
        pars_align_w_rng(&leaf_info1, 1.0, &leaf_info2, 1.0, &scoring, |_| 0);
    assert_eq!(score, 3.5);
    assert_eq!(alignment.map_x.len(), 4);
    assert_eq!(alignment.map_y.len(), 4);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 - -1));
}

#[test]
pub(crate) fn align_two_on_tree() {
    let mismatch = 1.0;
    let gap_open = 2.0;
    let gap_ext = 0.5;

    let sequences = Sequences::new(vec![
        Record::with_attrs("A", None, b"AACT"),
        Record::with_attrs("B", None, b"AC"),
    ]);
    let tree = tree!("(A:1.0, B:1.0):0.0;");
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let (alignment_vec, score) = pars_align_on_tree(&scoring, &tree, sequences);
    assert_eq!(score[Into::<usize>::into(tree.root)], 3.5);
    let alignment = &alignment_vec[&tree.root];
    assert_eq!(alignment.map_x.len(), 4);
    assert_eq!(alignment.map_y.len(), 4);
}

#[test]
pub(crate) fn internal_alignment_first_outcome() {
    let mismatch = 1.0;
    let gap_open = 2.0;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let leaf_info1 = [
        (vec![b'A'], NoGap),
        (vec![b'C', b'A'], NoGap),
        (vec![b'C'], GapOpen),
        (vec![b'T'], GapOpen),
    ]
    .map(create_site_info);

    let leaf_info2 = [([b'G'], GapOpen), ([b'A'], NoGap)].map(create_site_info);

    let (_, alignment, score) =
        pars_align_w_rng(&leaf_info1, 1.0, &leaf_info2, 1.0, &scoring, |_| 0);
    assert_eq!(score, 1.0);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - -));
}

#[allow(dead_code)]
pub(crate) fn create_site_info(args: (impl IntoIterator<Item = u8>, SiteFlag)) -> ParsimonySite {
    ParsimonySite::new(args.0, args.1)
}

#[test]
pub(crate) fn internal_alignment_second_outcome() {
    let mismatch = 1.0;
    let gap_open = 2.0;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let leaf_info1 = [
        (vec![b'A'], NoGap),
        (vec![b'A'], GapOpen),
        (vec![b'C'], GapOpen),
        (vec![b'T', b'C'], NoGap),
    ]
    .map(create_site_info);

    let leaf_info2 = [([b'G'], GapOpen), ([b'A'], NoGap)].map(create_site_info);

    let (_info, alignment, score) =
        pars_align_w_rng(&leaf_info1, 1.0, &leaf_info2, 1.0, &scoring, |_| 0);
    assert_eq!(score, 2.0);
    assert_eq!(alignment.map_x, align!(0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 - -1));
}

#[test]
pub(crate) fn internal_alignment_third_outcome() {
    let mismatch = 1.0;
    let gap_open = 2.0;
    let gap_ext = 0.5;
    let scoring = ParsimonyCostsSimple::new(mismatch, gap_open, gap_ext, dna());

    let leaf_info1 = [
        (vec![b'A'], NoGap),
        (vec![b'A'], GapOpen),
        (vec![b'C'], GapOpen),
        (vec![b'C', b'T'], NoGap),
    ]
    .map(create_site_info);

    let leaf_info2 = [(vec![b'G'], GapOpen), (vec![b'A'], NoGap)].map(create_site_info);

    let (_info, alignment, score) =
        pars_align_w_rng(&leaf_info1, 1.0, &leaf_info2, 1.0, &scoring, |l| l - 1);
    assert_eq!(score, 2.0);
    assert_eq!(alignment.map_x, align!(- 0 1 2 3));
    assert_eq!(alignment.map_y, align!(0 1 - - -));
}

#[test]
fn align_four_on_tree() {
    let a = 2.0;
    let b = 0.5;
    let c = 1.0;

    let sequences = Sequences::new(vec![
        rec!("A", b"AACT"),
        rec!("B", b"AC"),
        rec!("C", b"A"),
        rec!("D", b"GA"),
    ]);

    let tree = tree!("((A:1.0, B:1.0):1.0, (C:1.0, D:1.0):1.0);");
    let scoring = ParsimonyCostsSimple::new(c, a, b, dna());

    let (alignment, score) = pars_align_on_tree(&scoring, &tree, sequences);
    // first cherry
    let idx = &tree.by_id("A").parent.unwrap();
    assert_eq!(score[usize::from(idx)], 3.5);
    assert_eq!(alignment[idx].map_x.len(), 4);

    // second cherry
    let idx = &tree.by_id("C").parent.unwrap();
    assert_eq!(score[usize::from(idx)], 2.0);
    assert_eq!(alignment[idx].map_x.len(), 2);

    // root, three possible alignments
    let idx = &tree.root;
    assert!(score[usize::from(idx)] == 1.0 || score[usize::from(idx)] == 2.0);
    if score[2] == 1.0 {
        assert_eq!(alignment[idx].map_x.len(), 4);
    } else {
        assert!(alignment[idx].map_x.len() == 4 || alignment[idx].map_x.len() == 5);
    }
}
