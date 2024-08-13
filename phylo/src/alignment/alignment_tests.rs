use bio::io::fasta::Record;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::alignment::{
    sequences::Sequences, AlignmentBuilder, LeafMapping, NodeMapping, PairwiseAlignment as PA,
};
use crate::tree::tree_parser;
use crate::tree::{
    NodeIdx::{Internal as I, Leaf as L},
    Tree,
};

fn assert_alignment_eq(msa: &[Record], msa2: &[Record]) {
    for rec in msa2.iter() {
        let pos = msa.iter().position(|r| r.id() == rec.id()).unwrap();
        assert_eq!(&msa[pos], rec);
    }
}

#[cfg(test)]
fn aligned_seqs(ids: &[&str]) -> Sequences {
    Sequences::new(
        [
            Record::with_attrs("A0", Some("A0 sequence w 5 nucls"), b"AAAAA"),
            Record::with_attrs("B1", Some("B1 sequence w 1 nucl "), b"---A-"),
            Record::with_attrs("C2", Some("C2 sequence w 2 nucls"), b"AA---"),
            Record::with_attrs("D3", Some("D3 sequence w 1 nucl "), b"---A-"),
            Record::with_attrs("E4", Some("E4 sequence w 3 nucls"), b"-A-AA"),
        ]
        .into_iter()
        .filter(|rec| ids.contains(&rec.id()))
        .collect(),
    )
}

#[cfg(test)]
fn tree() -> Tree {
    tree_parser::from_newick_string(
        "((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;",
    )
    .unwrap()
    .pop()
    .unwrap()
}

#[cfg(test)]
fn unaligned_seqs() -> Sequences {
    let unaligned_seqs = aligned_seqs(&["A0", "B1", "C2", "D3", "E4"])
        .s
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
    Sequences::new(unaligned_seqs)
}

#[cfg(test)]
fn maps() -> (NodeMapping, LeafMapping) {
    let aligned_seqs = aligned_seqs(&["A0", "B1", "C2", "D3", "E4"]);
    (
        NodeMapping::from([
            (I(0), PA::new(align!(b"01234"), align!(b"01-23"))),
            (I(1), PA::new(align!(b"01234"), align!(b"---0-"))),
            (I(4), PA::new(align!(b"01--"), align!(b"-012"))),
            (I(6), PA::new(align!(b"-0-"), align!(b"012"))),
        ]),
        LeafMapping::from([
            (L(2), align!(aligned_seqs.s[0].seq())),
            (L(3), align!(aligned_seqs.s[1].seq())),
            (L(5), align!(aligned_seqs.s[2].seq())),
            (L(7), align!(aligned_seqs.s[3].seq())),
            (L(8), align!(aligned_seqs.s[4].seq())),
        ]),
    )
}

#[test]
fn sequences_from_aligned() {
    let seqs = vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"AAAAAA"),
        Record::with_attrs("B1", Some("B1 sequence"), b"---A-A"),
        Record::with_attrs("C2", Some("C2 sequence"), b"AA---A"),
        Record::with_attrs("D3", Some("D3 sequence"), b"---A-A"),
        Record::with_attrs("E4", Some("E4 sequence"), b"-A-AAA"),
    ];
    let sequences = Sequences::new(seqs.clone());
    assert_eq!(sequences.len(), 5);
    assert!(!sequences.is_empty());
    assert_eq!(sequences.msa_len(), 6);
    assert!(sequences.aligned);
    for (i, rec) in seqs.iter().enumerate() {
        assert_eq!(sequences.get(i), rec);
    }
}

#[test]
fn sequences_from_unaligned() {
    let seqs = vec![
        Record::with_attrs("A0", Some("A0 sequence"), b"AAAAAA"),
        Record::with_attrs("B1", Some("B1 sequence"), b"AA"),
        Record::with_attrs("C2", Some("C2 sequence"), b"AAA"),
        Record::with_attrs("D3", Some("D3 sequence"), b"AA"),
        Record::with_attrs("E4", Some("E4 sequence"), b"AAAA"),
    ];
    let sequences = Sequences::new(seqs.clone());
    assert_eq!(sequences.len(), 5);
    assert!(!sequences.is_empty());
    assert_eq!(sequences.msa_len(), 0);
    assert!(!sequences.aligned);
    for (i, rec) in seqs.iter().enumerate() {
        assert_eq!(sequences.get(i), rec);
    }
}

#[test]
fn sequences_from_empty() {
    let sequences = Sequences::new(vec![]);
    assert_eq!(sequences.len(), 0);
    assert!(sequences.is_empty());
    assert_eq!(sequences.msa_len(), 0);
    assert!(sequences.aligned);
}

#[test]
fn build_from_map() {
    let tree = tree();
    let unaligned_seqs = unaligned_seqs();
    let (node_map, leaf_map) = maps();
    let msa = AlignmentBuilder::new(&tree, unaligned_seqs.clone())
        .msa(node_map.clone())
        .build()
        .unwrap();
    assert_eq!(msa.node_map, node_map);
    assert_eq!(msa.leaf_map, leaf_map);
    assert_eq!(msa.seqs, unaligned_seqs);
}

#[test]
fn build_from_aligned_sequences() {
    let tree = tree();
    let unaligned_seqs = unaligned_seqs();
    let aligned_seqs = aligned_seqs(&["A0", "B1", "C2", "D3", "E4"]);
    let (node_map, leaf_map) = maps();
    let msa = AlignmentBuilder::new(&tree, aligned_seqs).build().unwrap();
    assert_eq!(msa.node_map, node_map);
    assert_eq!(msa.leaf_map, leaf_map);
    assert_eq!(msa.seqs, unaligned_seqs);
}

#[test]
fn different_build_compare() {
    let tree = tree();
    let unaligned_seqs = unaligned_seqs();
    let (node_map, _) = maps();
    let msa = AlignmentBuilder::new(&tree, unaligned_seqs)
        .msa(node_map.clone())
        .build()
        .unwrap();
    let aligned_seqs = aligned_seqs(&["A0", "B1", "C2", "D3", "E4"]);
    let msa2 = AlignmentBuilder::new(&tree, aligned_seqs).build().unwrap();
    assert_eq!(msa.node_map, msa2.node_map);
    assert_eq!(msa.leaf_map, msa2.leaf_map);
    assert_eq!(msa.seqs, msa2.seqs);
    assert_alignment_eq(
        &msa.compile(None, &tree).unwrap(),
        &msa2.compile(None, &tree).unwrap(),
    );
    assert_alignment_eq(
        &msa.compile(Some(I(4)), &tree).unwrap(),
        &msa2.compile(Some(I(4)), &tree).unwrap(),
    )
}

#[test]
fn compile_msa_root() {
    let tree = tree();
    let aligned_seqs = aligned_seqs(
        &["A0", "B1", "C2", "D3", "E4"]
            .into_iter()
            .choose_multiple(&mut thread_rng(), 5),
    );
    let msa = AlignmentBuilder::new(&tree, aligned_seqs.clone())
        .build()
        .unwrap();
    assert_alignment_eq(&msa.compile(None, &tree).unwrap(), &aligned_seqs.s);
}

#[test]
fn compile_msa_int1() {
    let tree = tree();
    let unaligned_seqs = unaligned_seqs();
    let (node_map, _) = maps();
    let msa = AlignmentBuilder::new(&tree, unaligned_seqs)
        .msa(node_map.clone())
        .build()
        .unwrap();
    let idx = tree.idx("I5").unwrap();
    assert_alignment_eq(
        &msa.compile(Some(I(idx)), &tree).unwrap(),
        &(aligned_seqs(&["A0", "B1"])).s,
    );
}

#[test]
fn compile_msa_int2() {
    let tree = tree();
    let unaligned_seqs = unaligned_seqs();
    let (node_map, _) = maps();
    let msa = AlignmentBuilder::new(&tree, unaligned_seqs)
        .msa(node_map.clone())
        .build()
        .unwrap();
    let idx = tree.idx("I6").unwrap();
    let d3 = aligned_seqs(&["D3"]).s.pop().unwrap();
    let e4 = aligned_seqs(&["E4"]).s.pop().unwrap();
    let data = vec![
        Record::with_attrs(d3.id(), d3.desc(), b"-A-"),
        Record::with_attrs(e4.id(), e4.desc(), b"AAA"),
    ];
    assert_alignment_eq(&msa.compile(Some(I(idx)), &tree).unwrap(), &data);
}

#[test]
fn compile_msa_leaf() {
    let tree = tree();
    let unaligned_seqs = unaligned_seqs();
    let (node_map, _) = maps();
    let msa = AlignmentBuilder::new(&tree, unaligned_seqs.clone())
        .msa(node_map.clone())
        .build()
        .unwrap();
    for leaf_id in tree.leaf_ids() {
        assert_alignment_eq(
            &msa.compile(Some(L(tree.idx(&leaf_id).unwrap())), &tree)
                .unwrap(),
            &[unaligned_seqs.get_by_id(&leaf_id).clone()],
        );
    }
}
