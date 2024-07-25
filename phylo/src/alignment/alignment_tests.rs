use std::collections::HashMap;

use crate::alignment::{compile_alignment_representation, Alignment};
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::tree::NodeIdx;
use crate::tree::{NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree};
use bio::io::fasta::Record;

macro_rules! align {
    (@collect -) => { None };
    (@collect $l:tt) => { Some($l) };
    ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
}

#[cfg(test)]
fn setup_test_tree() -> (PhyloInfo, HashMap<usize, Alignment>) {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(5, &L(0), &L(1), 1.0, 1.0);
    tree.add_parent(6, &L(3), &L(4), 1.0, 1.0);
    tree.add_parent(7, &L(2), &I(6), 1.0, 1.0);
    tree.add_parent(8, &I(5), &I(7), 1.0, 1.0);

    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();

    let info = PhyloInfo::from_sequences_tree(sequences, tree, &GapHandling::Ambiguous).unwrap();
    // ((0:1.0, 1:1.0)5:1.0,(2:1.0,(3:1.0, 4:1.0)6:1.0)7:1.0)8:1.0;
    let alignment = HashMap::<usize, Alignment>::from([
        (5, Alignment::new(align!(0 1 2 3 4), align!(- - - 0 -))),
        (6, Alignment::new(align!(- 0 -), align!(0 1 2))),
        (7, Alignment::new(align!(0 1 - -), align!(- 0 1 2))),
        (8, Alignment::new(align!(0 1 2 3 4), align!(0 1 - 2 3))),
    ]);
    // A0> AAAAA
    // B1> ---A-
    // C2> AA---
    // D3> ---A-
    // E4> -A-AA
    (info, alignment)
}

#[test]
fn alignment_compile_root() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, None::<NodeIdx>).unwrap();
    let data = [
        ("A0", "AAAAA"),
        ("B1", "---A-"),
        ("C2", "AA---"),
        ("D3", "---A-"),
        ("E4", "-A-AA"),
    ];
    for (i, (id, seq)) in data.iter().enumerate() {
        assert_eq!(msa[i].id(), *id);
        assert_eq!(msa[i].seq(), seq.as_bytes());
    }
}

#[test]
fn alignment_compile_internal1() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, Some(I(5))).unwrap();
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
}

#[test]
fn alignment_compile_internal2() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, Some(I(6))).unwrap();
    assert_eq!(msa[0].seq(), "-A-".as_bytes());
    assert_eq!(msa[1].seq(), "AAA".as_bytes());
}

#[test]
fn alignment_compile_leaf() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, Some(L(0))).unwrap();
    assert_eq!(msa.len(), 1);
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[0].id(), "A0");

    let msa = compile_alignment_representation(&info, &alignment, Some(L(1))).unwrap();
    assert_eq!(msa.len(), 1);
    assert_eq!(msa[0].seq(), "A".as_bytes());
    assert_eq!(msa[0].id(), "B1");

    let msa = compile_alignment_representation(&info, &alignment, Some(L(4))).unwrap();
    assert_eq!(msa.len(), 1);
    assert_eq!(msa[0].seq(), "AAA".as_bytes());
    assert_eq!(msa[0].id(), "E4")
}
