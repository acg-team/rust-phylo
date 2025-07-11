use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::bail;
use log::{info, warn};

use crate::alignment::{Aligner, Alignment, Sequences};
use crate::alphabets::Alphabet;
use crate::io::{self, DataError};
use crate::parsimony::ParsimonyAligner;
use crate::phylo_info::PhyloInfo;
use crate::tree::{build_nj_tree, Tree};
use crate::Result;

pub struct PhyloInfoBuilder {
    sequence_file: PathBuf,
    tree_file: Option<PathBuf>,
    alignment_builder: Option<Box<dyn Aligner>>,
    alphabet: Option<Alphabet>,
}

impl PhyloInfoBuilder {
    /// Creates a new empty PhyloInfoBuilder struct with only the sequence file path set.
    /// The tree file path is set to None.
    ///
    /// # Arguments
    /// * `sequence_file` - File path to the sequence fasta file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"));
    /// ```
    pub fn new(sequence_file: PathBuf) -> PhyloInfoBuilder {
        PhyloInfoBuilder {
            sequence_file,
            tree_file: None,
            alignment_builder: None,
            alphabet: None,
        }
    }

    /// Creates a new PhyloInfoBuilder struct with the sequence and tree file paths set.
    ///
    /// # Arguments
    /// * `sequence_file` - File path to the sequence fasta file.
    /// * `tree_file` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA_small.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"));
    /// ```
    pub fn with_attrs(sequence_file: PathBuf, tree_file: PathBuf) -> PhyloInfoBuilder {
        PhyloInfoBuilder {
            sequence_file,
            tree_file: Some(tree_file),
            alignment_builder: None,
            alphabet: None,
        }
    }

    /// Sets the tree file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the tree file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"))
    ///   .tree_file(Some(PathBuf::from("./data/tree_diff_branch_lengths_2.newick")));
    /// ```
    pub fn tree_file(mut self, path: Option<PathBuf>) -> PhyloInfoBuilder {
        self.tree_file = path;
        self
    }

    /// Sets the tree file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the tree file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the tree newick file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::alphabets::protein_alphabet;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let info = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta")).alphabet(Some(protein_alphabet())).build().unwrap();
    /// assert_eq!(info.msa.alphabet(), &protein_alphabet());
    /// ```
    pub fn alphabet(mut self, alphabet: Option<Alphabet>) -> PhyloInfoBuilder {
        self.alphabet = alphabet;
        self
    }

    /// Builds the PhyloInfo struct from the sequence file and the tree file (if provided).
    /// If the provided tree file has more than one tree, only the first tree will be processed.
    /// If no tree file is provided, an NJ tree is built from the sequences.
    /// Bails if no sequences are provided.
    /// Bails if the IDs of the tree leaves and the sequences do not match.
    /// Bails if the sequences are not aligned.
    /// Returns the PhyloInfo struct with the model type, tree, msa and leaf encoding set.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA_small.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"))
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(info.msa.len(), 8);
    /// assert_eq!(info.msa.seq_count(), 4);
    /// assert_eq!(info.tree.leaves().len(), 4);
    /// assert_eq!(info.tree.len(), 7);
    /// ```
    pub fn build(self) -> Result<PhyloInfo> {
        info!(
            "Reading sequences from file {}",
            self.sequence_file.display()
        );
        let sequences = if self.alphabet.is_none() {
            info!("No alphabet provided, detecting alphabet from sequences");
            Sequences::new(io::read_sequences(&self.sequence_file)?)
        } else {
            info!(
                "Using provided {} alphabet",
                self.alphabet.as_ref().unwrap()
            );
            Sequences::with_alphabet(
                io::read_sequences(&self.sequence_file)?,
                self.alphabet.unwrap(),
            )
        };
        info!("{} sequence(s) read successfully", sequences.len());

        let tree = match &self.tree_file {
            Some(tree_file) => self.read_tree(&sequences, tree_file)?,
            None => {
                info!("Building NJ tree from sequences");
                build_nj_tree(&sequences)?
            }
        };

        let msa = if sequences.aligned {
            info!("Sequences are aligned");
            Alignment::from_aligned(sequences, &tree)?
        } else {
            info!("Sequences are not aligned, aligning");
            self.alignment_builder
                .unwrap_or(Box::new(ParsimonyAligner::default()))
                .align(&sequences, &tree)?
        };

        Ok(PhyloInfo { tree, msa })
    }

    fn read_tree(&self, sequences: &Sequences, tree_file: &PathBuf) -> Result<Tree> {
        info!("Reading trees from file {}", tree_file.display());
        let mut trees = io::read_newick_from_file(tree_file)?;
        info!("{} tree(s) read successfully", trees.len());
        self.check_tree_number(&trees)?;
        let tree = trees.remove(0);
        self.validate_taxa_ids(&tree, sequences)?;
        Ok(tree)
    }

    /// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
    fn validate_taxa_ids(&self, tree: &Tree, sequences: &Sequences) -> Result<()> {
        let tip_ids: HashSet<String> = HashSet::from_iter(tree.leaf_ids());
        let sequence_ids: HashSet<String> =
            HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
        info!("Checking that tree tip and sequence IDs match");
        let mut missing_tips = sequence_ids.difference(&tip_ids).collect::<Vec<_>>();
        if !missing_tips.is_empty() {
            missing_tips.sort();
            bail!(DataError {
                message: format!("Mismatched IDs found, missing tree tip IDs: {missing_tips:?}")
            });
        }
        let mut missing_seqs = tip_ids.difference(&sequence_ids).collect::<Vec<_>>();
        if !missing_seqs.is_empty() {
            missing_seqs.sort();
            bail!(DataError {
                message: format!("Mismatched IDs found, missing sequence IDs: {missing_seqs:?}")
            });
        }
        Ok(())
    }

    /// Checks that there is at least one tree in the vector, bails with an error otherwise.
    /// Prints a warning if there is more than one tree because only the first tree will be processed.
    fn check_tree_number(&self, trees: &[Tree]) -> Result<()> {
        if trees.is_empty() {
            bail!(DataError {
                message: String::from("No trees in the tree file, aborting")
            });
        }
        if trees.len() > 1 {
            warn!("More than one tree in the tree file, only the first tree will be processed");
        }
        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
pub mod private_tests {
    use std::path::PathBuf;

    use super::PhyloInfoBuilder as PIB;

    #[test]
    fn builder_setters() {
        let fasta1 = PathBuf::from("./data/sequences_DNA_small.fasta");
        let newick = PathBuf::from("./data/tree_diff_branch_lengths_2.newick");

        let builder = PIB::new(fasta1.clone());
        assert_eq!(builder.sequence_file, fasta1);
        assert_eq!(builder.tree_file, None);
        let builder = builder.tree_file(Some(newick.clone()));
        assert_eq!(builder.tree_file, Some(newick));
        let builder = builder.tree_file(None);
        assert_eq!(builder.tree_file, None);
    }
}
