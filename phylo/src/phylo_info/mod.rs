use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::bail;
use bio::io::fasta::Record;
use log::{info, warn};

use crate::io::{self, DataError};
use crate::tree::Tree;
use crate::Result;

/// The PhyloInfo struct contains all the information needed to run a phylogenetic analysis.
///
/// # TODO:
/// * Add getters/setters for the fields, make the fields private.
#[derive(Debug)]
pub struct PhyloInfo {
    /// Unaligned phylogenetic sequences.
    pub sequences: Vec<Record>,
    /// Multiple sequence alignment of the sequences, if they are aligned.
    pub msa: Option<Vec<Record>>,
    /// Phylogenetic tree.
    pub tree: Tree,
}

/// Converts the given sequences to uppercase and returns a new vector.
fn make_sequences_uppercase(sequences: &[Record]) -> Vec<Record> {
    sequences
        .iter()
        .map(|rec| Record::with_attrs(rec.id(), rec.desc(), &rec.seq().to_ascii_uppercase()))
        .collect()
}

/// Creates a PhyloInfo struct from a vector of fasta records and a tree.
/// The sequences might not be aligned, but the ids of the tree leaves and provided sequences must match.
/// In the output the sequences are sorted by the leaf ids and converted to uppercase.
///
/// # Arguments
/// * `sequences` - Vector of fasta records.
/// * `tree` - Phylogenetic tree.
///
/// # Example
/// ```
/// # use bio::io::fasta::Record;
/// # use phylo::tree::Tree;
/// # fn make_test_data() -> (Vec<Record>, Tree) {
/// #   use phylo::tree::NodeIdx::{Internal as I, Leaf as L};
/// #   let sequences = vec![
/// #       Record::with_attrs("A", None, b"aaaa"),
/// #       Record::with_attrs("B", None, b"cccc"),
/// #       Record::with_attrs("C", None, b"gg"),
/// #       Record::with_attrs("D", None, b"TTTTTTT"),
/// #   ];
/// #   let mut tree = Tree::new(&sequences).unwrap();
/// #   tree.add_parent(0, L(0), L(1), 2.0, 2.0);
/// #   tree.add_parent(1, I(0), L(2), 1.0, 2.0);
/// #   tree.add_parent(2, I(1), L(3), 1.0, 2.0);
/// #   tree.complete = true;
/// #   tree.create_postorder();
/// #   tree.create_preorder();
/// #   (sequences, tree)
/// # }
/// use phylo::phylo_info::phyloinfo_from_sequences_tree;
/// let (sequences, tree) = make_test_data();
/// let info = phyloinfo_from_sequences_tree(&sequences, tree).unwrap();
/// assert!(info.msa.is_none());
/// for (i, node) in info.tree.leaves.iter().enumerate() {
///     assert!(info.sequences[i].id() == node.id);
/// }
/// for rec in info.sequences.iter() {
///     assert!(!rec.seq().is_empty());
///     assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
/// }
/// ```
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

/// Creates a PhyloInfo struct from a two given files, one containing the sequences in fasta format and
/// one containing the tree in newick format.
/// The sequences might not be aligned, but the ids of the tree leaves and provided sequences must match.
/// In the output the sequences are sorted by the leaf ids and converted to uppercase.
///
/// # Arguments
/// * `sequence_file` - File path to the sequence fasta file.
/// * `tree_file` - File path to the tree newick file.
///
/// # Example
/// ```
/// use std::path::PathBuf;
/// use phylo::phylo_info::phyloinfo_from_files;
/// let info = phyloinfo_from_files(
///     PathBuf::from("./data/sequences_DNA_small.fasta"),
///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick")).unwrap();
/// assert!(info.msa.is_some());
/// for (i, node) in info.tree.leaves.iter().enumerate() {
///     assert!(info.sequences[i].id() == node.id);
/// }
/// for rec in info.sequences.iter() {
///     assert!(!rec.seq().is_empty());
///     assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
/// }
/// ```
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

/// Returns a vector of records representing the MSA if all the sequences are of the same length.
/// Otherwise returns None.
///
/// # Arguments
/// * `sequences` - Vector of fasta records.
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

/// Checks that there are sequences in the vector, bails with an error otherwise.
fn check_sequences_not_empty(sequences: &Vec<Record>) -> Result<()> {
    if sequences.is_empty() {
        bail!(DataError {
            message: String::from("No sequences provided, aborting.")
        });
    }
    Ok(())
}

/// Sorts sequences in the vector to match the order of the leaves in the tree.
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

/// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
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

/// Checks that there is at least one tree in the vector, bails with an error otherwise.
/// Prints a warning if there is more than one tree because only the first tree will be processed.
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
