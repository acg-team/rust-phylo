use std::path::PathBuf;

use rand::seq::IteratorRandom;
use rand::thread_rng;

use crate::alignment::{
    sequences::Sequences, InternalAlignments, PairwiseAlignment as PA, SeqMaps,
};
use crate::alignment::{Alignment, AncestralAlignment, MASA, MSA};
use crate::alphabets::{
    dna_alphabet, protein_alphabet, AMINOACIDS, GAP, NUCLEOTIDES, NUCLEOTIDE_INDEX,
};
use crate::io::read_sequences;
use crate::phylo_info::{self, PhyloInfo};
use crate::tree::{
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
fn test_tree() -> Tree {
    tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;")
}

#[cfg(test)]
fn maps() -> (InternalAlignments, SeqMaps) {
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    (
        InternalAlignments::from([
            (I(0), PA::new(align!(b"01234"), align!(b"01-23"))),
            (I(1), PA::new(align!(b"01234"), align!(b"---0-"))),
            (I(4), PA::new(align!(b"01--"), align!(b"-012"))),
            (I(6), PA::new(align!(b"-0-"), align!(b"012"))),
        ]),
        SeqMaps::from([
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
    assert_ne!(dna_seqs.alphabet().symbols(), AMINOACIDS);
    assert_eq!(*dna_seqs.alphabet(), dna_alphabet());

    let protein_seqs = Sequences::with_alphabet(records.clone(), protein_alphabet());
    assert_eq!(protein_seqs.alphabet().symbols(), AMINOACIDS);
    assert_ne!(protein_seqs.alphabet().symbols(), NUCLEOTIDES);
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
    let msa = MSA::from_aligned(aligned_seqs.clone(), &tree).unwrap();

    assert_eq!(*msa.internal_alignments(), node_map);
    assert_eq!(msa.leaf_maps, leaf_map);
    assert_eq!(msa.seqs, unaligned_seqs);
    assert_eq!(msa.len(), 5);
    assert_eq!(
        PhyloInfo { msa, tree }.compile_alignment(None).unwrap(),
        aligned_seqs
    );
}

#[test]
fn fail_from_unaligned_sequences() {
    let seqs = Sequences::new(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"AA"),
        record!("C2", Some("C2 sequence"), b"AAA"),
        record!("D3", Some("D3 sequence"), b"AA"),
        record!("E4", Some("E4 sequence"), b"AAAA"),
    ]);
    let tree = test_tree();
    let msa = MSA::from_aligned(seqs, &tree);
    assert!(msa.is_err());
}

#[test]
fn compile_msa_root() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(
        &["A0", "B1", "C2", "D3", "E4"]
            .into_iter()
            .choose_multiple(&mut thread_rng(), 5),
    );
    let msa = MSA::from_aligned(aligned_seqs.clone(), &tree).unwrap();
    assert_eq!(
        PhyloInfo { msa, tree }.compile_alignment(None).unwrap(),
        aligned_seqs
    );
}

#[test]
fn compile_msa_int1() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let msa = MSA::from_aligned(aligned_seqs, &tree).unwrap();
    let node_idx = tree.idx("I5");
    assert_eq!(
        PhyloInfo { msa, tree }
            .compile_alignment(Some(&node_idx))
            .unwrap(),
        test_alignment(&["A0", "B1"]),
    );
}

#[test]
fn compile_msa_int2() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let msa = MSA::from_aligned(aligned_seqs, &tree).unwrap();
    let d3 = test_alignment(&["D3"]).s.pop().unwrap();
    let e4 = test_alignment(&["E4"]).s.pop().unwrap();
    let node_idx = tree.idx("I6");
    let data = Sequences::new(vec![
        record!(d3.id(), d3.desc(), b"-A-"),
        record!(e4.id(), e4.desc(), b"AAA"),
    ]);
    assert_eq!(
        PhyloInfo { msa, tree }
            .compile_alignment(Some(&node_idx))
            .unwrap(),
        data
    );
}

#[test]
fn compile_msa_leaf() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let unaligned_seqs = aligned_seqs.clone().into_gapless();
    let msa = MSA::from_aligned(aligned_seqs.clone(), &tree).unwrap();
    let phylo_info = PhyloInfo { msa, tree };
    for leaf_id in phylo_info.tree.leaf_ids() {
        assert_eq!(
            phylo_info
                .compile_alignment(Some(&phylo_info.tree.idx(&leaf_id)))
                .unwrap(),
            Sequences::new(vec![unaligned_seqs.record_by_id(&leaf_id).clone()])
        );
    }
}

#[test]
fn input_msa_empty_col() {
    let tree = test_tree();
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_empty_col.fasta")).unwrap());
    let msa = MSA::from_aligned(sequences, &tree).unwrap();
    assert_eq!(msa.len(), 40 - 1);
    assert_eq!(msa.seq_count(), 5);
}

#[test]
fn display_sequences() {
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap());
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
        read_sequences(&PathBuf::from("./data/sequences_DNA2_unaligned.fasta")).unwrap(),
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

// TODO: perhaps we want the default fmt to be like with msa.compile instead of printing the
// unaligned seqs.
// see: https://github.com/acg-team/rust-phylo/issues/62
#[test]
fn fmt_alignment() {
    // arrange
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap());
    let msa = MSA::from_aligned(sequences, &tree).unwrap();
    let true_content_with_gaps = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let true_content = true_content_with_gaps.replace(GAP as char, "");
    let mut true_lines = true_content.lines().collect::<Vec<_>>();
    true_lines.sort();

    // act
    let s = format!("{}", msa);

    // assert
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);
    assert_eq!(lines, true_lines);
}

#[test]
fn display_alignment() {
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap());
    let msa = MSA::from_aligned(sequences, &tree).unwrap();

    let s = format!("{}", msa);
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();

    assert_eq!(lines, true_lines);
}

#[test]
fn display_ancestral_alignment() {
    // arrange
    let tree = tree!("((C:0.1,D:0.2)I01:0.3,(A:0.4,B:0.5)I02:0.6)Root;");
    let sequences = Sequences::new(
        read_sequences(&PathBuf::from("./data/sequences_DNA1_with_ancestors.fasta")).unwrap(),
    );
    let msa = MASA::from_aligned_with_ancestral(sequences, &tree).unwrap();
    let true_lines = std::fs::read_to_string("./data/sequences_DNA1_with_ancestors.fasta").unwrap();
    let mut true_lines = true_lines.lines().collect::<Vec<_>>();
    true_lines.sort();

    // act
    let lines = format!("{}", msa);

    // assert
    let mut lines = lines.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 14);
    assert_eq!(lines, true_lines);
}

#[test]
fn masa_compile_root() {
    // arrange
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap());
    let masa = MASA::from_aligned(sequences.clone(), &tree).unwrap();
    let phylo_info = PhyloInfo { msa: masa, tree };

    // act
    let compiled = phylo_info.compile_alignment(None).unwrap();

    // assert
    assert_eq!(compiled, sequences);
}

#[test]
fn masa_compile_subroot() {
    // arrange
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new(read_sequences(&PathBuf::from("./data/sequences_DNA1.fasta")).unwrap());
    let masa = MASA::from_aligned(sequences.clone(), &tree).unwrap();
    let subroot_id = "C";
    let true_compiled = Sequences::new(
        sequences
            .iter()
            .filter(|f| f.id() == subroot_id)
            .cloned()
            .collect(),
    );
    let phylo_info = PhyloInfo { msa: masa, tree };

    // act
    let compiled = phylo_info
        .compile_alignment(Some(&phylo_info.tree.idx(subroot_id)))
        .unwrap();
    let true_compiled = true_compiled.into_gapless();

    // assert
    assert_eq!(compiled, true_compiled);
}

#[test]
fn removing_gap_cols() {
    let mut seqs = Sequences::new(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ]);
    seqs.remove_gap_cols();
    assert_eq!(seqs.len(), 5);
    for seq in seqs.iter() {
        assert_eq!(seq.seq().len(), 6);
    }

    let mut seqs2 = Sequences::new(vec![
        record!("A0", Some("A0 sequence"), b"-AAA-AA-A"),
        record!("B1", Some("B1 sequence"), b"-----A--A"),
        record!("C2", Some("C2 sequence"), b"-AA-----A"),
        record!("D3", Some("D3 sequence"), b"-----A--A"),
        record!("E4", Some("E4 sequence"), b"--A--AA-A"),
    ]);
    for seq in seqs2.iter() {
        assert_eq!(seq.seq().len(), 9);
    }

    seqs2.remove_gap_cols();
    assert_eq!(seqs2.len(), 5);
    for seq in seqs2.iter() {
        assert_eq!(seq.seq().len(), 6);
    }
    assert_eq!(seqs, seqs2)
}

#[test]
#[should_panic]
fn removing_gap_cols_on_unaligned() {
    let mut seqs = Sequences::new(vec![
        record!("A0", None, b"AAAAAA"),
        record!("B1", None, b"AA"),
        record!("C2", None, b"AAA"),
        record!("D3", None, b"AA"),
        record!("E4", None, b"AAAA"),
    ]);
    seqs.remove_gap_cols();
}
