use std::collections::HashMap;

use bio::io::fasta::Record;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::alignment::{Alignment, AlignmentBuilder, PairwiseAlignment as PA};
use crate::phylo_info::{GapHandling, PhyloInfo, PhyloInfoBuilder};
use crate::tree::tree_parser;
use crate::tree::{
    NodeIdx::{self, Internal as I, Leaf as L},
    Tree,
};

fn assert_alignment_eq(msa: Vec<Record>, data: &[Record]) {
    for rec in data.iter() {
        let pos = msa.iter().position(|r| r.id() == rec.id()).unwrap();
        assert_eq!(&msa[pos], rec);
    }
}

#[cfg(test)]
fn final_alignment(ids: &[&str]) -> Vec<Record> {
    [
        Record::with_attrs("A0", Some("A0 sequence w 5 nucls"), b"AAAAA"),
        Record::with_attrs("B1", Some("B1 sequence w 1 nucl "), b"---A-"),
        Record::with_attrs("C2", Some("C2 sequence w 2 nucls"), b"AA---"),
        Record::with_attrs("D3", Some("D3 sequence w 1 nucl "), b"---A-"),
        Record::with_attrs("E4", Some("E4 sequence w 3 nucls"), b"-A-AA"),
    ]
    .into_iter()
    .filter(|rec| ids.contains(&rec.id()))
    .collect()
}

#[cfg(test)]
fn test_info() -> PhyloInfo {
    let tree = tree_parser::from_newick_string(
        "((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;",
    )
    .unwrap()
    .pop()
    .unwrap();
    let msa = final_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let sequences = msa
        .into_iter()
        .map(|rec| {
            Record::with_attrs(
                rec.id(),
                rec.desc(),
                &rec.seq()
                    .iter()
                    .filter(|&c| c != &b'-')
                    .copied()
                    .collect::<Vec<u8>>(),
            )
        })
        .choose_multiple(&mut thread_rng(), 5);
    PhyloInfoBuilder::build_from_objects(sequences, tree, GapHandling::Ambiguous).unwrap()
}

#[cfg(test)]
fn alignment<'a>(tree: &'a Tree, sequences: &'a [Record]) -> Alignment<'a> {
    let msa = HashMap::<NodeIdx, PA>::from([
        (I(0), PA::new(align!(b"01234"), align!(b"01-23"))),
        (I(1), PA::new(align!(b"01234"), align!(b"---0-"))),
        (I(4), PA::new(align!(b"01--"), align!(b"-012"))),
        (I(6), PA::new(align!(b"-0-"), align!(b"012"))),
    ]);
    let leaf_maps = sequences.iter().map(|rec| align!(rec.seq())).collect();
    Alignment {
        tree,
        msa,
        leaf_maps,
        sequences: sequences.to_vec(),
    }
}

#[test]
fn compile_root() {
    let info = test_info();
    let msa = alignment(&info.tree, &info.sequences);
    assert_alignment_eq(
        msa.compile(None),
        &final_alignment(&["A0", "B1", "C2", "D3", "E4"]),
    );
}

#[test]
fn compile_int1() {
    let info = test_info();
    let msa = alignment(&info.tree, &info.sequences);
    let idx = info.tree.idx("I5").unwrap();
    assert_alignment_eq(msa.compile(Some(I(idx))), &final_alignment(&["A0", "B1"]));
}

#[test]
fn compile_int2() {
    let info = test_info();
    let msa = alignment(&info.tree, &info.sequences);
    let idx = info.tree.idx("I6").unwrap();
    let d3 = final_alignment(&["D3"]).pop().unwrap();
    let e4 = final_alignment(&["E4"]).pop().unwrap();
    let data = vec![
        Record::with_attrs(d3.id(), d3.desc(), b"-A-"),
        Record::with_attrs(e4.id(), e4.desc(), b"AAA"),
    ];
    assert_alignment_eq(msa.compile(Some(I(idx))), &data);
}

#[test]
fn compile_leaf() {
    let info = test_info();
    let msa = alignment(&info.tree, &info.sequences);
    for leaf_id in info.tree.leaf_ids() {
        assert_alignment_eq(
            msa.compile(Some(L(info.tree.idx(&leaf_id).unwrap()))),
            &[info.sequence(&leaf_id).unwrap().clone()],
        );
    }
}

#[test]
fn align() {
    let info = test_info();
    let aligned_seqs = final_alignment(&["A0", "B1", "C2", "D3", "E4"])
        .into_iter()
        .choose_multiple(&mut thread_rng(), 5);
    let msa = AlignmentBuilder::with_attrs(&info.tree, &aligned_seqs)
        .from_aligned_sequences()
        .unwrap();
    assert_alignment_eq(
        msa.compile(None),
        &final_alignment(&["A0", "B1", "C2", "D3", "E4"]),
    );
}
