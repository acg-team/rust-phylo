use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::bail;
use bio::io::fasta::Record;
use log::{info, warn};

use crate::io::{self, DataError};
use crate::tree::{tree_parser, Tree};
use crate::Result;

#[derive(Debug)]
pub struct PhyloInfo {
    pub sequences: Vec<Record>,
    pub msa: Option<Vec<Record>>,
    pub tree: Tree,
}

fn make_sequences_uppercase(sequences: &[Record]) -> Vec<Record> {
    sequences
        .iter()
        .map(|rec| Record::with_attrs(rec.id(), rec.desc(), &rec.seq().to_ascii_uppercase()))
        .collect()
}

impl PhyloInfo {
pub fn phyloinfo_from_sequences_tree(sequences: &[Record], tree: Tree) -> Result<PhyloInfo> {
    let mut sequences = sequences.to_vec();
    check_sequences_not_empty(&sequences)?;
    sequences = make_sequences_uppercase(&sequences);

    validate_tree_sequence_ids(&tree, &sequences)?;
    sort_sequences_by_leaf_ids(&tree, &mut sequences);

    let msa = get_msa_if_aligned(&sequences);
    Ok(PhyloInfo {
        sequences,
        tree,
        msa,
    })
}

pub fn phyloinfo_from_sequences_newick(
    sequences: &[Record],
    newick_string: &str,
) -> Result<PhyloInfo> {
    let mut sequences = sequences.to_vec();
    check_sequences_not_empty(&sequences)?;
    sequences = make_sequences_uppercase(&sequences);
    let mut trees = tree_parser::from_newick_string(newick_string)?;
    info!("{} tree(s) parsed successfully", trees.len());

    check_tree_number(&trees)?;
    let tree = trees.remove(0);
    validate_tree_sequence_ids(&tree, &sequences)?;
    sort_sequences_by_leaf_ids(&tree, &mut sequences);

    let msa = get_msa_if_aligned(&sequences);
    Ok(PhyloInfo {
        sequences,
        tree,
        msa,
    })
}

pub fn phyloinfo_from_files(sequence_file: PathBuf, tree_file: PathBuf) -> Result<PhyloInfo> {
    info!("Reading sequences from file {}", sequence_file.display());
    let mut sequences = io::read_sequences_from_file(sequence_file)?;
    info!("{} sequence(s) read successfully", sequences.len());
    check_sequences_not_empty(&sequences)?;
    sequences = make_sequences_uppercase(&sequences);

    info!("Reading trees from file {}", tree_file.display());
    let mut trees = io::read_newick_from_file(tree_file)?;
    info!("{} tree(s) read successfully", trees.len());

    check_tree_number(&trees)?;
    let tree = trees.remove(0);

    validate_tree_sequence_ids(&tree, &sequences)?;
    sort_sequences_by_leaf_ids(&tree, &mut sequences);

    let msa = get_msa_if_aligned(&sequences);
    Ok(PhyloInfo {
        sequences,
        tree,
        msa,
    })
}

fn get_msa_if_aligned(sequences: &[Record]) -> Option<Vec<Record>> {
    let sequence_length = sequences[0].seq().len();
    if sequences
        .iter()
        .filter(|rec| rec.seq().len() != sequence_length)
        .count()
        == 0
    {
        Some(sequences.to_vec())
    } else {
        None
    }
}

fn check_sequences_not_empty(sequences: &Vec<Record>) -> Result<()> {
    if sequences.is_empty() {
        bail!(DataError {
            message: String::from("No sequences provided, aborting.")
        });
    }
    Ok(())
}

fn sort_sequences_by_leaf_ids(tree: &Tree, sequences: &mut [Record]) {
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
}

fn validate_tree_sequence_ids(tree: &Tree, sequences: &[Record]) -> Result<()> {
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.get_leaf_ids());
    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
    info!("Checking that tree tip and sequence IDs match");
    if (&tip_ids | &sequence_ids) != (&tip_ids & &sequence_ids) {
        let discrepancy: HashSet<_> = tip_ids.symmetric_difference(&sequence_ids).collect();
        bail!(DataError {
            message: format!("Mismatched IDs found: {:?}", discrepancy)
        });
    }
    Ok(())
}

fn check_tree_number(trees: &Vec<Tree>) -> Result<()> {
    if trees.is_empty() {
        bail!(DataError {
            message: String::from("No trees in the tree file, aborting.")
        });
    }

    if trees.len() > 1 {
        warn!("More than one tree in the tree file, only the first tree will be processed.");
    }

    Ok(())
}

#[cfg(test)]
mod phylo_info_tests;
