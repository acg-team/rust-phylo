use anyhow::bail;
use log::{info, warn};
use std::collections::HashSet;
use std::path::PathBuf;

use bio::io::fasta::Record;

use crate::tree::Tree;
use crate::{io, Result};

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
        sequence_file.to_str().unwrap()
    );
    let sequences = io::read_sequences_from_file(sequence_file)?;
    info!("{} sequence(s) read successfully", sequences.len());

    let sequence_ids: HashSet<String> =
        HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
    info!("Reading trees from file {}", tree_file.to_str().unwrap());
    let mut trees = io::read_newick_from_file(tree_file)?;
    info!("{} tree(s) read successfully", trees.len());

    if trees.is_empty() {
        bail!("No trees in the tree file, aborting.");
        todo!("Make our own tree from the sequences.");
    }
    
    if trees.len() > 1 {
        warn!("More than one tree in the tree file, only the first tree will be processed.");
    }

    let tree = trees.pop().unwrap();
    let tip_ids: HashSet<String> = HashSet::from_iter(tree.get_leaf_ids());
    info!("Checking that tree tip and sequence IDs match");
    if (&tip_ids | &sequence_ids) != (&tip_ids & &sequence_ids) {
        let discrepancy: HashSet<_> = tip_ids.symmetric_difference(&sequence_ids).collect();
        bail!("Mismatched IDs found: {:?}", discrepancy);
    }

    Ok(PhyloInfo { sequences, tree })
}
