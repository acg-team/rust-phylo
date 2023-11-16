use std::{fmt::Debug, fmt::Display, path::PathBuf};

use assert_matches::assert_matches;
use bio::io::fasta::Record;

use crate::io::DataError;
use crate::phylo_info::{
    get_msa_if_aligned, phyloinfo_from_files, phyloinfo_from_sequences_newick,
    phyloinfo_from_sequences_tree, PhyloInfo,
};
use crate::tree::tree_parser::ParsingError;
use crate::tree::NodeIdx::{Internal as I, Leaf as L};
use crate::tree::Tree;

#[test]
fn setup_info_correct() {
    let res_info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .unwrap();
    assert_eq!(res_info.tree.leaves.len(), 4);
    assert_eq!(res_info.sequences.len(), 4);
}

#[test]
fn setup_info_mismatched_ids() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "Mismatched IDs found: {\"C\", \"D\"}" | "Mismatched IDs found: {\"D\", \"C\"}"
    );
}

#[test]
fn setup_info_missing_sequence_file() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA_nonexistent.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    );
    assert_matches!(
        info.unwrap_err().to_string().as_str(),
        "Failed to read fasta from \"./data/sequences_DNA_nonexistent.fasta\""
    );
}

#[test]
fn setup_info_empty_sequence_file() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_empty.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No sequences provided, aborting."
    );
}

#[test]
fn setup_info_empty_tree_file() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_empty.newick"),
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No trees in the tree file, aborting."
    );
}

#[test]
fn setup_info_malformed_tree_file() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_malformed.newick"),
    );
    assert!(downcast_error::<ParsingError>(&info)
        .to_string()
        .contains("Malformed newick string"));
}

#[test]
fn setup_info_multiple_trees() {
    let res_info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_multiple.newick"),
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
    let info = phyloinfo_from_files(
        PathBuf::from("./data/real_examples/HIV_subset.fas"),
        PathBuf::from("./data/real_examples/HIV_subset.nwk"),
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
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .unwrap();
    assert!(info.msa.is_none())
}

#[test]
fn setup_aligned_msa() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sequences_DNA1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
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
    assert!(get_msa_if_aligned(&sequences).is_none());
    let sequences = vec![
        Record::with_attrs("A0", None, b"AAAAA"),
        Record::with_attrs("B1", None, b"A----"),
        Record::with_attrs("C2", None, b"AABCD"),
        Record::with_attrs("D3", None, b"AAAAA"),
        Record::with_attrs("E4", None, b"AAATT"),
    ];
    assert!(get_msa_if_aligned(&sequences).is_some());
}

#[test]
fn check_phyloinfo_creation_newick_msa() {
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATATAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ];
    let info = phyloinfo_from_sequences_newick(&sequences, "((A:2.0,B:2.0):1.0,C:2.0):0.0;");
    assert!(info.is_ok());
    assert!(info.unwrap().msa.is_some());
}

#[test]
fn check_phyloinfo_creation_newick_no_msa() {
    let sequences = vec![
        Record::with_attrs("A", None, b"CTATATAAC"),
        Record::with_attrs("B", None, b"ATATATATAA"),
        Record::with_attrs("C", None, b"TTATATATAT"),
    ];
    let info = phyloinfo_from_sequences_newick(&sequences, "((A:2.0,B:2.0):1.0,C:2.0):0.0;");
    assert!(info.is_ok());
    assert!(info.unwrap().msa.is_none());
}

#[test]
fn check_phyloinfo_creation_newick_no_seqs() {
    let info = phyloinfo_from_sequences_newick(&[], "((A:2.0,B:2.0):1.0,C:2.0):0.0;");
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_newick_mismatch_ids() {
    let sequences = vec![
        Record::with_attrs("D", None, b"CTATATAAC"),
        Record::with_attrs("E", None, b"ATATATATAA"),
        Record::with_attrs("F", None, b"TTATATATAT"),
    ];
    let info = phyloinfo_from_sequences_newick(&sequences, "((A:2.0,B:2.0):1.0,C:2.0):0.0;");
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
    let mut tree = Tree::new(&sequences);
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
    let info = phyloinfo_from_sequences_tree(
        &vec![
            Record::with_attrs("A", None, b"AAAAA"),
            Record::with_attrs("B", None, b"A"),
            Record::with_attrs("C", None, b"AA"),
            Record::with_attrs("D", None, b"A"),
        ],
        make_test_tree(),
    );
    assert!(info.is_ok());
    assert!(info.unwrap().msa.is_none());
}

#[test]
fn check_phyloinfo_creation_tree_correct_msa() {
    let info = phyloinfo_from_sequences_tree(
        &vec![
            Record::with_attrs("A", None, b"AA"),
            Record::with_attrs("B", None, b"A-"),
            Record::with_attrs("C", None, b"AA"),
            Record::with_attrs("D", None, b"A-"),
        ],
        make_test_tree(),
    );
    assert!(info.is_ok());
    assert!(info.unwrap().msa.is_some());
}

#[test]
fn check_phyloinfo_creation_tree_mismatch_ids() {
    let info = phyloinfo_from_sequences_tree(
        &vec![
            Record::with_attrs("D", None, b"CTATATAAC"),
            Record::with_attrs("E", None, b"ATATATATAA"),
            Record::with_attrs("F", None, b"TTATATATAT"),
        ],
        make_test_tree(),
    );
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_tree_no_seqs() {
    let info = phyloinfo_from_sequences_tree(&[], make_test_tree());
    assert!(info.is_err());
}
