use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::bail;
use log::{info, warn};

use crate::alignment::{AlignmentBuilder, Sequences};
use crate::io::{self, DataError};
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::tree::{build_nj_tree, Tree};
use crate::Result;

pub struct PhyloInfoBuilder {
    sequence_file: PathBuf,
    tree_file: Option<PathBuf>,
    gap_handling: GapHandling,
}

impl PhyloInfoBuilder {
    /// Creates a new empty PhyloInfoBuilder struct with only the sequence file path set.
    /// The tree file path is set to None and and gap handling is set to Proper.
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
            gap_handling: GapHandling::Proper,
        }
    }

    /// Creates a new PhyloInfoBuilder struct with the sequence file path, tree file path and gap handling set.
    ///
    /// # Arguments
    /// * `sequence_file` - File path to the sequence fasta file.
    /// * `tree_file` - File path to the tree newick file.
    /// * `gap_handling` - Gap handling option -- treat gaps as ambiguous characters or properly.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::{PhyloInfoBuilder, GapHandling};
    /// let builder = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA_small.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    ///     GapHandling::Ambiguous);
    /// ```
    pub fn with_attrs(
        sequence_file: PathBuf,
        tree_file: PathBuf,
        gap_handling: GapHandling,
    ) -> PhyloInfoBuilder {
        PhyloInfoBuilder {
            sequence_file,
            tree_file: Some(tree_file),
            gap_handling,
        }
    }

    /// Sets the sequence file path for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the sequence file path set.
    ///
    /// # Arguments
    /// * `path` - File path to the sequence fasta file.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"))
    ///    .sequence_file(PathBuf::from("./data/sequences_DNA_small.fasta"));
    /// ```
    pub fn sequence_file(mut self, path: PathBuf) -> PhyloInfoBuilder {
        self.sequence_file = path;
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
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"))
    ///   .tree_file(PathBuf::from("./data/tree_diff_branch_lengths_2.newick"));
    /// ```
    pub fn tree_file(mut self, path: PathBuf) -> PhyloInfoBuilder {
        self.tree_file = Some(path);
        self
    }

    /// Sets the gap handling option for the PhyloInfoBuilder struct.
    /// Returns the PhyloInfoBuilder struct with the gap handling option set.
    ///
    /// # Arguments
    /// * `gap_handling` - Gap handling option -- treat gaps as ambiguous characters or properly.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::{GapHandling, PhyloInfoBuilder};
    /// let builder = PhyloInfoBuilder::new(PathBuf::from("./data/sequences_DNA_small.fasta"))
    ///     .gap_handling(GapHandling::Ambiguous);
    /// ```
    pub fn gap_handling(mut self, gap_handling: GapHandling) -> PhyloInfoBuilder {
        self.gap_handling = gap_handling;
        self
    }

    /// Builds the PhyloInfo struct from the sequence file and the tree file (if provided).
    /// If the provided tree file has more than one tree, only the first tree will be processed.
    /// If no tree file is provided, an NJ tree is built from the sequences.
    /// Bails if no sequences are provided.
    /// Bails if the IDs of the tree leaves and the sequences do not match.
    /// Bails if the sequences are not aligned.
    /// Returns the PhyloInfo struct with the model type, tree, msa, gap handling and leaf encoding set.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::phylo_info::{GapHandling, PhyloInfoBuilder};
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA_small.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    ///     GapHandling::Ambiguous)
    ///     .build()
    ///     .unwrap();
    /// assert_eq!(info.msa.msa_len(), 8);
    /// assert_eq!(info.msa.len(), 4);
    /// assert_eq!(info.tree.leaves().len(), 4);
    /// assert_eq!(info.tree.len(), 7);
    /// ```
    pub fn build(self) -> Result<PhyloInfo> {
        info!(
            "Reading sequences from file {}",
            self.sequence_file.display()
        );
        let sequences = Sequences::with_attrs(
            io::read_sequences_from_file(&self.sequence_file)?,
            &self.gap_handling,
        );
        info!("{} sequence(s) read successfully", sequences.len());

        let tree = match &self.tree_file {
            Some(tree_file) => {
                info!("Reading trees from file {}", tree_file.display());
                let mut trees = io::read_newick_from_file(tree_file)?;
                info!("{} tree(s) read successfully", trees.len());
                Self::check_tree_number(&trees)?;
                trees.remove(0)
            }
            None => {
                info!("Building NJ tree from sequences");
                build_nj_tree(&sequences)?
            }
        };
        Self::build_from_objects(sequences, tree)
    }

    /// Builds the PhyloInfo struct from the provided sequences and tree.
    /// Bails if no sequences are provided.
    /// Bails if the IDs of the tree leaves and the sequences do not match.
    /// Bails if the sequences are not aligned.
    /// Returns the PhyloInfo struct with the model type, tree, msa, gap handling and leaf encoding set.
    pub(crate) fn build_from_objects(sequences: Sequences, tree: Tree) -> Result<PhyloInfo> {
        if sequences.is_empty() {
            bail!(DataError {
                message: String::from("No sequences provided, aborting.")
            });
        }
        Self::validate_tree_sequence_ids(&tree, &sequences)?;
        let msa = AlignmentBuilder::new(&tree, sequences).build()?;
        // let leaf_encoding = Self::leaf_encoding(&msa.seqs, &gap_handling);
        let mut info = PhyloInfo {
            tree: tree.clone(),
            msa,
            leaf_encoding: HashMap::new(),
        };
        info.generate_leaf_encoding();
        Ok(info)
    }

    /// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
    fn validate_tree_sequence_ids(tree: &Tree, sequences: &Sequences) -> Result<()> {
        let tip_ids: HashSet<String> = HashSet::from_iter(tree.leaf_ids());
        let sequence_ids: HashSet<String> =
            HashSet::from_iter(sequences.iter().map(|rec| rec.id().to_string()));
        info!("Checking that tree tip and sequence IDs match.");
        let mut missing_tips = sequence_ids.difference(&tip_ids).collect::<Vec<_>>();
        if !missing_tips.is_empty() {
            missing_tips.sort();
            bail!(DataError {
                message: format!(
                    "Mismatched IDs found, missing tree tip IDs: {:?}",
                    missing_tips
                )
            });
        }
        let mut missing_seqs = tip_ids.difference(&sequence_ids).collect::<Vec<_>>();
        if !missing_seqs.is_empty() {
            missing_seqs.sort();
            bail!(DataError {
                message: format!(
                    "Mismatched IDs found, missing sequence IDs: {:?}",
                    missing_seqs
                )
            });
        }
        Ok(())
    }

    /// Checks that there is at least one tree in the vector, bails with an error otherwise.
    /// Prints a warning if there is more than one tree because only the first tree will be processed.
    fn check_tree_number(trees: &[Tree]) -> Result<()> {
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
}
