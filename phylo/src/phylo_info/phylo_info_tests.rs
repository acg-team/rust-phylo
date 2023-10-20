use super::{setup_phylogenetic_info, PhyloInfo};
use crate::io::DataError;
use crate::tree::tree_parser::ParsingError;
use assert_matches::assert_matches;
use std::{fmt::Debug, fmt::Display, path::PathBuf};

#[test]
fn setup_info_correct() {
    let res_info = setup_phylogenetic_info(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .unwrap();
    assert_eq!(res_info.tree.leaves.len(), 4);
    assert_eq!(res_info.sequences.len(), 4);
}

#[test]
fn setup_info_mismatched_ids() {
    let info = setup_phylogenetic_info(
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
    let info = setup_phylogenetic_info(
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
    let info = setup_phylogenetic_info(
        PathBuf::from("./data/sequences_empty.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_1.newick"),
    );
    assert_matches!(
        downcast_error::<DataError>(&info).to_string().as_str(),
        "No sequences in the file, aborting."
    );
}

#[test]
fn setup_info_empty_tree_file() {
    let info = setup_phylogenetic_info(
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
    let info = setup_phylogenetic_info(
        PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
        PathBuf::from("./data/tree_malformed.newick"),
    );
    assert!(downcast_error::<ParsingError>(&info)
        .to_string()
        .contains("Malformed newick string"));
}

#[test]
fn setup_info_multiple_trees() {
    let res_info = setup_phylogenetic_info(
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
    let info = setup_phylogenetic_info(
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
