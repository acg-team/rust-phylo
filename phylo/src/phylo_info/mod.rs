use crate::io::{self, DataError};
use crate::tree::Tree;
use crate::Result;
use anyhow::bail;
use bio::io::fasta::Record;
use log::{info, warn};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug)]
pub struct PhyloInfo {
    pub sequences: Vec<Record>,
    pub tree: Tree,
}

impl PhyloInfo {
    pub fn new(tree: Tree, sequences: Vec<Record>) -> Self {
        PhyloInfo { sequences, tree }
    }
}

pub fn setup_phylogenetic_info(sequence_file: PathBuf, tree_file: PathBuf) -> Result<PhyloInfo> {
    info!(
        "Reading unaligned sequences from file {}",
        sequence_file.display()
    );
    let mut sequences = io::read_sequences_from_file(sequence_file)?;
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

    let tree = trees.remove(0);
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.get_leaf_ids());
    info!("Checking that tree tip and sequence IDs match");
    if (&tip_ids | &sequence_ids) != (&tip_ids & &sequence_ids) {
        let discrepancy: HashSet<_> = tip_ids.symmetric_difference(&sequence_ids).collect();
        bail!(DataError {
            message: format!("Mismatched IDs found: {:?}", discrepancy)
        });
    }
    let id_index: HashMap<&str, usize> = tree
        .leaves
        .iter()
        .enumerate()
        .map(|(index, leaf)| (leaf.id.as_str(), index))
        .collect();
    sequences.sort_by_key(|record| {
        id_index
            .get(record.id())
            .cloned()
            .unwrap_or(std::usize::MAX)
    });
    Ok(PhyloInfo { sequences, tree })
}

#[cfg(test)]
mod phylo_info_tests;
