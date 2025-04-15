use std::path::PathBuf;

use bio::io::fasta::Record;
use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::alignment::{
    sequences::Sequences, AlignmentBuilder, AncestralAlignmentBuilder, InternalMapping,
    LeafMapping, PairwiseAlignment as PA,
};
use crate::alphabets::{dna_alphabet, protein_alphabet, AMINOACIDS, NUCLEOTIDES};
use crate::io::read_sequences_from_file;
use crate::tree::{
    tree_parser::from_newick,
    NodeIdx::{Internal as I, Leaf as L},
    Tree,
};
use crate::{align, record, tree};

#[cfg(test)]
fn test_alignment(ids: &[&str]) -> Sequences {
    Sequences::new(
        [
            record!("A0", Some("A0 sequence w 5 nucls"), b"AAAAA"),
            record!("B1", Some("B1 sequence w 1 nucl "), b"---A-"),
            record!("C2", Some("C2 sequence w 2 nucls"), b"AA---"),
            record!("D3", Some("D3 sequence w 1 nucl "), b"---A-"),
            record!("E4", Some("E4 sequence w 3 nucls"), b"-A-AA"),
        ]
        .into_iter()
        .filter(|rec| ids.contains(&rec.id()))
        .collect(),
    )
}

#[cfg(test)]
fn test_alignment_with_ancestors() -> Sequences {
    test_alignment_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4", "I5", "I6", "I7", "I8"])
}

#[cfg(test)]
fn test_alignment_with_ancestors_subset(ids: &[&str]) -> Sequences {
    Sequences::new(
        [
            record!("A0", Some("A0 sequence w 5 nucls"), b"A--AAA"),
            record!("B1", Some("B1 sequence w 1 nucl "), b"-A--AA"),
            record!("C2", Some("C2 sequence w 2 nucls"), b"A-A-A-"),
            record!("D3", Some("D3 sequence w 1 nucl "), b"-A-A--"),
            record!("E4", Some("E4 sequence w 3 nucls"), b"--A---"),
            record!("I5", Some("I5 sequence w 3 nucls"), b"XX-XXX"),
            record!("I6", Some("I6 sequence w 3 nucls"), b"-XXX--"),
            record!("I7", Some("I7 sequence w 3 nucls"), b"XXXXX-"),
            record!("I8", Some("I8 sequence w 3 nucls"), b"XX-XX-"),
        ]
        .into_iter()
        .filter(|rec| ids.contains(&rec.id()))
        .collect(),
    )
}

/// Returns this tree:
/// ```text
///         I8
///        /  \
///       /    \
///     I5      I7
///    /  \    /  \
///  A0    B1 C2   I6
///                / \
///               D3  E4
/// ```
#[cfg(test)]
fn test_tree() -> Tree {
    tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;")
}

#[cfg(test)]
fn maps() -> (InternalMapping, LeafMapping) {
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    (
        InternalMapping::from([
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

#[cfg(test)]
fn maps_with_ancestors() -> LeafMapping {
    let aligned_seqs = test_alignment_with_ancestors();
    LeafMapping::from([
        (I(0), align!(aligned_seqs.s[8].seq())),
        (I(1), align!(aligned_seqs.s[5].seq())),
        (L(2), align!(aligned_seqs.s[0].seq())),
        (L(3), align!(aligned_seqs.s[1].seq())),
        (I(4), align!(aligned_seqs.s[7].seq())),
        (L(5), align!(aligned_seqs.s[2].seq())),
        (I(6), align!(aligned_seqs.s[6].seq())),
        (L(7), align!(aligned_seqs.s[3].seq())),
        (L(8), align!(aligned_seqs.s[4].seq())),
    ])
}

#[test]
fn sequences_from_aligned() {
    let seqs = vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ];
    let sequences = Sequences::new(seqs.clone());
    assert_eq!(sequences.len(), 5);
    assert!(!sequences.is_empty());
    assert!(sequences.aligned);
    for (i, rec) in seqs.iter().enumerate() {
        assert_eq!(sequences.record(i), rec);
    }
}

#[test]
fn sequences_from_unaligned() {
    let seqs = vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"AA"),
        record!("C2", Some("C2 sequence"), b"AAA"),
        record!("D3", Some("D3 sequence"), b"AA"),
        record!("E4", Some("E4 sequence"), b"AAAA"),
    ];
    let sequences = Sequences::new(seqs.clone());
    assert_eq!(sequences.len(), 5);
    assert!(!sequences.is_empty());
    assert!(!sequences.aligned);
    for (i, rec) in seqs.iter().enumerate() {
        assert_eq!(sequences.record(i), rec);
    }
}

#[test]
fn sequences_from_empty() {
    let sequences = Sequences::new(vec![]);
    assert_eq!(sequences.len(), 0);
    assert!(sequences.is_empty());
    assert!(sequences.aligned);
}

#[test]
fn sequences_with_alphabet() {
    let records = vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ];

    let dna_seqs = Sequences::with_alphabet(records.clone(), dna_alphabet());
    assert_eq!(dna_seqs.alphabet().symbols(), NUCLEOTIDES);
    assert_eq!(*dna_seqs.alphabet(), dna_alphabet());

    let protein_seqs = Sequences::with_alphabet(records.clone(), protein_alphabet());
    assert_eq!(protein_seqs.alphabet().symbols(), AMINOACIDS);
    assert_eq!(*protein_seqs.alphabet(), protein_alphabet());
}

#[test]
fn sequences_into_gapless() {
    // arrange
    let expected_seqs = Sequences::new(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"AA"),
        record!("C2", Some("C2 sequence"), b"AAA"),
        record!("D3", Some("D3 sequence"), b"AA"),
        record!("E4", Some("E4 sequence"), b"AAAA"),
    ]);

    // act
    let gapless_seqs = Sequences::new(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ])
    .into_gapless();

    // assert
    assert_eq!(gapless_seqs.len(), 5);
    assert!(!gapless_seqs.is_empty());
    assert!(!gapless_seqs.aligned);
    assert_eq!(gapless_seqs, expected_seqs);
}

#[test]
fn build_from_aligned_sequences() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let unaligned_seqs = aligned_seqs.into_gapless();
    let (node_map, leaf_map) = maps();
    let msa = AlignmentBuilder::new(&tree, aligned_seqs.clone())
        .build()
        .unwrap();
    assert_eq!(msa.node_map, node_map);
    assert_eq!(msa.leaf_map, leaf_map);
    assert_eq!(msa.seqs, unaligned_seqs);
    assert_eq!(msa.len(), 5);
    assert_eq!(msa.compile(None, &tree).unwrap(), aligned_seqs);
}

#[test]
fn compile_msa_root() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(
        &["A0", "B1", "C2", "D3", "E4"]
            .into_iter()
            .choose_multiple(&mut thread_rng(), 5),
    );
    let msa = AlignmentBuilder::new(&tree, aligned_seqs.clone())
        .build()
        .unwrap();
    assert_eq!(msa.compile(None, &tree).unwrap(), aligned_seqs);
}

#[test]
fn compile_msa_int1() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let msa = AlignmentBuilder::new(&tree, aligned_seqs).build().unwrap();
    assert_eq!(
        msa.compile(Some(&tree.idx("I5")), &tree).unwrap(),
        test_alignment(&["A0", "B1"]),
    );
}

#[test]
fn compile_msa_int2() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let msa = AlignmentBuilder::new(&tree, aligned_seqs).build().unwrap();
    let d3 = test_alignment(&["D3"]).s.pop().unwrap();
    let e4 = test_alignment(&["E4"]).s.pop().unwrap();
    let data = Sequences::new(vec![
        record!(d3.id(), d3.desc(), b"-A-"),
        record!(e4.id(), e4.desc(), b"AAA"),
    ]);
    assert_eq!(msa.compile(Some(&tree.idx("I6")), &tree).unwrap(), data);
}

#[test]
fn compile_msa_leaf() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let unaligned_seqs = aligned_seqs.clone().into_gapless();
    let msa = AlignmentBuilder::new(&tree, aligned_seqs.clone())
        .build()
        .unwrap();
    for leaf_id in tree.leaf_ids() {
        assert_eq!(
            msa.compile(Some(&tree.idx(&leaf_id)), &tree).unwrap(),
            Sequences::new(vec![unaligned_seqs.record_by_id(&leaf_id).clone()])
        );
    }
}

#[test]
fn input_msa_empty_col() {
    let tree = test_tree();
    let sequences = Sequences::new(
        read_sequences_from_file(&PathBuf::from("./data/sequences_empty_col.fasta")).unwrap(),
    );
    let msa = AlignmentBuilder::new(&tree, sequences).build().unwrap();
    assert_eq!(msa.len(), 40 - 1);
    assert_eq!(msa.seq_count(), 5);
}

#[test]
fn display_sequences() {
    let sequences = Sequences::new(
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap(),
    );
    let s = format!("{}", sequences);
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();
    assert_eq!(lines, true_lines);
}

#[test]
fn display_unaligned_sequences() {
    let sequences = Sequences::new(
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA2_unaligned.fasta")).unwrap(),
    );
    let s = format!("{}", sequences);
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA2_unaligned.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();
    assert_eq!(lines, true_lines);
}

#[test]
fn display_alignment() {
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences = Sequences::new(
        read_sequences_from_file(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap(),
    );
    let msa = AlignmentBuilder::new(&tree, sequences).build().unwrap();

    let s = format!("{}", msa.compile(None, &tree).unwrap());
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();

    assert_eq!(lines, true_lines);
}

#[test]
fn build_ancestral_alignment_from_wrong_number_nodes() {
    // arrange
    let tree = test_tree();
    let seqs = test_alignment_with_ancestors_subset(&["A0"]);

    // act
    let msa = AncestralAlignmentBuilder::new(&tree, seqs).build();

    // assert
    match msa {
        Ok(_) => panic!("Expected Err, but got Ok"),
        Err(msg) => {
            assert!(msg.contains("The number of sequences does not match the number of nodes"))
        }
    }
}

#[test]
fn build_ancestral_alignment_from_unaligned_fails() {
    // arrange
    let tree = test_tree();
    let mut seqs = test_alignment_with_ancestors();
    seqs.aligned = false;

    // act
    let msa = AncestralAlignmentBuilder::new(&tree, seqs).build();

    // assert
    match msa {
        Ok(_) => panic!("Expected Err, but got Ok"),
        Err(msg) => {
            assert!(msg.contains("Unaligned sequences are not yet supported"))
        }
    }
}

#[test]
fn build_ancestral_alignment_from_aligned_sequences() {
    // arrange
    let tree = test_tree();
    let seqs = test_alignment_with_ancestors();

    // act
    let msa = AncestralAlignmentBuilder::new(&tree, seqs).build().unwrap();
    let msa_len = msa.len();
    let seq_count = msa.seq_count();

    // assert
    assert_eq!(msa.seq_map, maps_with_ancestors(),);
    assert_eq!(msa.seqs.s, test_alignment_with_ancestors().into_gapless().s);
    assert_eq!(msa_len, 6);
    assert_eq!(seq_count, 9)
}

#[test]
fn build_ancestral_alignment_from_only_aligned_leaf_seqs() {
    // arrange
    let tree = test_tree();
    let seqs = test_alignment_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4"]);

    // act
    let msa = AncestralAlignmentBuilder::new(&tree, seqs).build().unwrap();
    let msa_len = msa.len();
    let seq_count = msa.seq_count();

    // assert
    assert_eq!(msa.seq_map, maps_with_ancestors(),);
    // TODO: these do not contain the ancestral wildcard seqs
    // assert_eq!(msa.seqs.s, test_alignment_with_ancestors().into_gapless().s);
    assert_eq!(msa_len, 6);
    assert_eq!(seq_count, 9)
}

#[test]
fn test_display_ancestral_alignment() {
    // arrange
    let tree = test_tree();
    let seqs = test_alignment_with_ancestors_subset(&["A0", "B1", "C2", "D3", "E4"]);
    let msa = AncestralAlignmentBuilder::new(&tree, seqs).build().unwrap();

    // act
    let s = format!("{}", msa);

    // assert
    let mut s = s.split("\n").collect::<Vec<_>>();
    s.sort();
    assert_eq!(s.len(), 15);
    assert_eq!(s[0].len(), 0); // since there is a newline at the end of every internal node line
    assert_eq!(s[1], ">A0 A0 sequence w 5 nucls");
    assert_eq!(s[2], ">B1 B1 sequence w 1 nucl ");
    assert_eq!(s[3], ">C2 C2 sequence w 2 nucls");
    assert_eq!(s[4], ">D3 D3 sequence w 1 nucl ");
    assert_eq!(s[5], ">E4 E4 sequence w 3 nucls");
    assert_eq!(s[6], "A");
    assert_eq!(s[7], "AA");
    assert_eq!(s[8], "AAA");
    assert_eq!(s[9], "AAA");
    assert_eq!(s[10], "AAAA");
    assert_eq!(s[11], "internal node 0, XX-XX-");
    assert_eq!(s[12], "internal node 1, XX-XXX");
    assert_eq!(s[13], "internal node 4, XXXXX-");
    assert_eq!(s[14], "internal node 6, -XXX--");
}
