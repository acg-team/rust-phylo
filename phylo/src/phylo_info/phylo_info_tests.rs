use std::{fmt::Debug, fmt::Display, path::PathBuf};

use assert_matches::assert_matches;
use bio::io::fasta::Record;

use crate::io::DataError;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::tree::tree_parser::{self, ParsingError};
use crate::tree::NodeIdx::{Internal as I, Leaf as L};
use crate::tree::Tree;

#[cfg(test)]
fn tree_newick(newick: &str) -> Tree {
    tree_parser::from_newick_string(newick)
        .unwrap()
        .pop()
        .unwrap()
}

#[test]
fn setup_info_correct() {
    let res_info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    assert_eq!(res_info.tree.leaves.len(), 4);
    assert_eq!(res_info.sequences.len(), 4);
    for (i, node) in res_info.tree.leaves.iter().enumerate() {
        assert!(res_info.sequences[i].id() == node.id);
    }
    for rec in res_info.sequences.iter() {
        assert!(!rec.seq().is_empty());
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    }
    assert_eq!(res_info.msa, None);
}

#[test]
fn setup_info_mismatched_ids() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
        &GapHandling::Ambiguous,
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "Mismatched IDs found: {\"C\", \"D\"}" | "Mismatched IDs found: {\"D\", \"C\"}"
    );
}

#[test]
fn setup_info_missing_sequence_file() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA_nonexistent.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
        &GapHandling::Ambiguous,
    );
    assert_matches!(
        info.unwrap_err().to_string().as_str(),
        "Failed to read fasta from \"./data/sequences_DNA_nonexistent.fasta\""
    );
}

#[test]
fn setup_info_empty_sequence_file() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_empty.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
        &GapHandling::Ambiguous,
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No sequences provided, aborting."
    );
}

#[test]
fn setup_info_empty_tree_file() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_empty.newick"),
        &GapHandling::Ambiguous,
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No trees in the tree file, aborting."
    );
}

#[test]
fn setup_info_malformed_tree_file() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_malformed.newick"),
        &GapHandling::Ambiguous,
    );
    assert!(downcast_error::<ParsingError>(&info)
        .to_string()
        .contains("Malformed newick string"));
}

#[test]
fn setup_info_multiple_trees() {
    let res_info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_multiple.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    assert_eq!(res_info.tree.leaves.len(), 4);
    assert_eq!(res_info.sequences.len(), 4);
}

fn downcast_error<T: Display + Debug + Send + Sync + 'static>(
    result: &Result<PhyloInfo, anyhow::Error>,
) -> &T {
    (result.as_ref().unwrap_err()).downcast_ref::<T>().unwrap()
}

#[test]
fn info_check_sequence_order() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/real_examples/HIV_subset.fas"),
        PathBuf::from("./data/real_examples/HIV_subset.nwk"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    for i in 0..info.sequences.len() {
        assert_eq!(
            info.sequences[i].id(),
            info.tree.leaves[i].id,
            "Sequences and tree leaves are not in the same order"
        );
    }
}

#[test]
fn setup_unaligned_empty_msa() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    assert!(info.msa.is_none())
}

#[test]
fn setup_aligned_msa() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sequences_DNA1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    assert!(info.msa.is_some());
    assert_eq!(info.msa.as_ref().unwrap(), info.sequences.as_slice());
}

#[test]
fn test_aligned_check() {
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A"),
        Record::with_attrs("C2", None, b"AA"),
        Record::with_attrs("D3", None, b"A"),
        Record::with_attrs("E4", None, b"AAA"),
    ];
    assert!(PhyloInfo::msa_if_aligned(&sequences).is_none());
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A----"),
        Record::with_attrs("C2", None, b"AABCD"),
        Record::with_attrs("D3", None, b"AAAAA"),
        Record::with_attrs("E4", None, b"AAATT"),
    ];
    assert!(PhyloInfo::msa_if_aligned(&sequences).is_some());
}

#[test]
fn check_phyloinfo_creation_newick_msa() {
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ];
    let info = PhyloInfo::from_sequences_tree(
        sequences,
        tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
        &GapHandling::Ambiguous,
    );
    assert!(info.is_ok());
    assert!(info.unwrap().msa.is_some());
}

#[test]
fn check_phyloinfo_creation_tree_no_msa() {
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATAAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ];
    let info = PhyloInfo::from_sequences_tree(
        sequences,
        tree_newick("((A:2.0,B:2.0):1.0,C:2.0):0.0;"),
        &GapHandling::Ambiguous,
    );
    let res_info = info.unwrap();
    assert!(res_info.msa.is_none());
    for (i, node) in res_info.tree.leaves.iter().enumerate() {
        assert!(res_info.sequences[i].id() == node.id);
    }
    for rec in res_info.sequences.iter() {
        assert!(!rec.seq().is_empty());
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    }
}

#[test]
fn check_phyloinfo_creation_tree_no_seqs() {
    let info = PhyloInfo::from_sequences_tree(
        vec![],
        tree_parser::from_newick_string("((A:2.0,B:2.0):1.0,C:2.0):0.0;")
            .unwrap()
            .pop()
            .unwrap(),
        &GapHandling::Ambiguous,
    );
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_newick_mismatch_ids() {
    let sequences = vec![
        Record::with_attrs("D", None, b"CTATATAAC"),
        Record::with_attrs("E", None, b"ATATATATAA"),
        Record::with_attrs("F", None, b"TTATATATAT"),
    ];
    let info = PhyloInfo::from_sequences_tree(
        sequences,
        tree_parser::from_newick_string("((A:2.0,B:2.0):1.0,C:2.0):0.0;")
            .unwrap()
            .pop()
            .unwrap(),
        &GapHandling::Ambiguous,
    );
    assert!(info.is_err());
}

#[cfg(test)]
fn make_test_tree() -> Tree {
    let sequences = vec![
        Record::with_attrs("A", None, b""),
        Record::with_attrs("B", None, b""),
        Record::with_attrs("C", None, b""),
        Record::with_attrs("D", None, b""),
    ];
    let mut tree = Tree::new(&sequences).unwrap();
    tree.add_parent(0, L(0), L(1), 2.0, 2.0);
    tree.add_parent(1, I(0), L(2), 1.0, 2.0);
    tree.add_parent(2, I(1), L(3), 1.0, 2.0);
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    tree
}

#[test]
fn check_phyloinfo_creation_tree_correct_no_msa() {
    let info = PhyloInfo::from_sequences_tree(
        vec![
            Record::with_attrs("A", None, b"AAAAA"),
            Record::with_attrs("B", None, b"A"),
            Record::with_attrs("C", None, b"AA"),
            Record::with_attrs("D", None, b"A"),
        ],
        make_test_tree(),
        &GapHandling::Ambiguous,
    );
    let res_info = info.unwrap();
    assert!(res_info.msa.is_none());
    for (i, node) in res_info.tree.leaves.iter().enumerate() {
        assert!(res_info.sequences[i].id() == node.id);
    }
    for rec in res_info.sequences.iter() {
        assert!(!rec.seq().is_empty());
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    }
}

#[test]
fn check_phyloinfo_creation_tree_correct_msa() {
    let info = PhyloInfo::from_sequences_tree(
        vec![
            Record::with_attrs("A", None, b"AA"),
            Record::with_attrs("B", None, b"A-"),
            Record::with_attrs("C", None, b"AA"),
            Record::with_attrs("D", None, b"A-"),
        ],
        make_test_tree(),
        &GapHandling::Ambiguous,
    );
    assert!(info.is_ok());
    assert!(info.unwrap().msa.is_some());
}

#[test]
fn check_phyloinfo_creation_tree_mismatch_ids() {
    let info = PhyloInfo::from_sequences_tree(
        vec![
            Record::with_attrs("D", None, b"CTATATAAC"),
            Record::with_attrs("E", None, b"ATATATATAA"),
            Record::with_attrs("F", None, b"TTATATATAT"),
        ],
        make_test_tree(),
        &GapHandling::Ambiguous,
    );
    assert!(info.is_err());
}

#[test]
fn check_empirical_frequencies() {
    let info = PhyloInfo::from_sequences_tree(
        vec![
            Record::with_attrs("A", None, b"AAAACCC"),
            Record::with_attrs("B", None, b"TTTCC"),
            Record::with_attrs("C", None, b"GGG"),
            Record::with_attrs("D", None, b"TTAAA"),
        ],
        make_test_tree(),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let counts = info.counts();
    assert_eq!(counts.clone().into_values().sum::<f64>(), 20.0);
    assert_eq!(counts[&b'A'], 7.0);
    assert_eq!(counts[&b'C'], 5.0);
    assert_eq!(counts[&b'G'], 3.0);
    assert_eq!(counts[&b'T'], 5.0);
    assert_eq!(counts[&b'-'], 0.0);
}

#[test]
fn check_phyloinfo_creation_sequences() {
    let sequences = vec![
        Record::with_attrs("C", None, b"TTATATATAT"),
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
    ];
    let info = PhyloInfo::from_sequences(sequences, &GapHandling::Ambiguous);
    assert!(info.is_ok());
    let info = info.unwrap();
    assert!(info.msa.is_some());
    assert_eq!(
        info.sequences
            .iter()
            .map(|rec| rec.id())
            .collect::<Vec<_>>(),
        info.tree
            .leaves
            .iter()
            .map(|node| node.id.clone())
            .collect::<Vec<_>>(),
    );
    assert!(info.tree.complete);
}
