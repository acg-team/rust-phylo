use anyhow::bail;
use log::{info, warn};
use std::collections::HashSet;
use std::path::PathBuf;

use bio::io::fasta::Record;

use crate::tree::Tree;
use crate::{io, Result};

use crate::io::DataError;

#[derive(Debug)]
pub(super) struct PhyloInfo {
    pub(super) sequences: Vec<Record>,
    pub(super) tree: Tree,
}

pub(super) fn setup_phylogenetic_info(
    sequence_file: PathBuf,
    tree_file: PathBuf,
) -> Result<PhyloInfo> {
    info!(
        "Reading unaligned sequences from file {}",
        sequence_file.display()
    );
    let sequences = io::read_sequences_from_file(sequence_file)?;
    info!("{} sequence(s) read successfully", sequences.len());
    if sequences.is_empty() {
        bail!(DataError {
            message: String::from("No sequences in the file, aborting.")
        });
    }

    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));

    info!("Reading trees from file {}", tree_file.display());
    let mut trees = io::read_newick_from_file(tree_file)?;
    info!("{} tree(s) read successfully", trees.len());

    if trees.is_empty() {
        bail!(DataError {
            message: String::from("No trees in the tree file, aborting.")
        });
    }

    if trees.len() > 1 {
        warn!("More than one tree in the tree file, only the first tree will be processed.");
    }

    let tree = trees.pop().unwrap();
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.get_leaf_ids());
    info!("Checking that tree tip and sequence IDs match");
    if (&tip_ids | &sequence_ids) != (&tip_ids & &sequence_ids) {
        let discrepancy: HashSet<_> = tip_ids.symmetric_difference(&sequence_ids).collect();
        bail!(DataError {
            message: String::from(format!("Mismatched IDs found: {:?}", discrepancy))
        });
    }

    Ok(PhyloInfo { sequences, tree })
}

#[cfg(test)]
mod phylo_info_tests {
    use super::setup_phylogenetic_info;
    use super::PhyloInfo;
    use crate::io::DataError;
    use crate::tree::ParsingError;
    use assert_matches::assert_matches;

    use std::{fmt::Debug, fmt::Display, path::PathBuf};

    #[test]
    fn setup_info_correct() {
        let res_info = setup_phylogenetic_info(
            PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
            PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
        ).unwrap();
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

    fn downcast_error<T: Display + Debug + Send + Sync + 'static>(
        result: &Result<PhyloInfo, anyhow::Error>,
    ) -> &T {
        (result.as_ref().unwrap_err()).downcast_ref::<T>().unwrap()
    }
}
