use super::{compile_alignment_representation, Alignment};
use crate::phylo_info::{phyloinfo_from_sequences_tree, GapHandling, PhyloInfo};
use crate::tree::NodeIdx;
use crate::tree::{NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree};
use bio::io::fasta::Record;

macro_rules! align {
    (@collect -) => { None };
    (@collect $l:tt) => { Some($l) };
    ( $( $e:tt )* ) => {vec![ $( align!(@collect $e), )* ]};
}

fn setup_test_tree() -> (PhyloInfo, Vec<Alignment>) {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(0, L(0), L(1), 1.0, 1.0);
    tree.add_parent(1, L(3), L(4), 1.0, 1.0);
    tree.add_parent(2, L(2), I(1), 1.0, 1.0);
    tree.add_parent(3, I(0), I(2), 1.0, 1.0);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();

    let info = phyloinfo_from_sequences_tree(sequences, tree, &GapHandling::Ambiguous).unwrap();
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
    (info, alignment)
}

#[test]
fn alignment_compile_root() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, None::<NodeIdx>);
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
    assert_eq!(msa[2].seq(), "AA---".as_bytes());
    assert_eq!(msa[3].seq(), "---A-".as_bytes());
    assert_eq!(msa[4].seq(), "-A-AA".as_bytes());
}

#[test]
fn alignment_compile_internal1() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, Some(I(0)));
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[1].seq(), "---A-".as_bytes());
}

#[test]
fn alignment_compile_internal2() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, Some(I(1)));
    assert_eq!(msa[0].seq(), "-A-".as_bytes());
    assert_eq!(msa[1].seq(), "AAA".as_bytes());
}

#[test]
fn alignment_compile_leaf() {
    let (info, alignment) = setup_test_tree();
    let msa = compile_alignment_representation(&info, &alignment, Some(L(0)));
    assert_eq!(msa.len(), 1);
    assert_eq!(msa[0].seq(), "AAAAA".as_bytes());
    assert_eq!(msa[0].id(), "A0");

    let msa = compile_alignment_representation(&info, &alignment, Some(L(1)));
    assert_eq!(msa.len(), 1);
    assert_eq!(msa[0].seq(), "A".as_bytes());
    assert_eq!(msa[0].id(), "B1");

    let msa = compile_alignment_representation(&info, &alignment, Some(L(4)));
    assert_eq!(msa.len(), 1);
    assert_eq!(msa[0].seq(), "AAA".as_bytes());
    assert_eq!(msa[0].id(), "E4")
}
