use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use anyhow::bail;
use bio::io::fasta::Record;
use log::{info, warn};
use nalgebra::DMatrix;

use crate::alphabets::{alphabet_from_type, sequence_type};
use crate::evolutionary_models::ModelType;
use crate::io::{self, DataError};
use crate::substitution_models::FreqVector;
use crate::tree::{build_nj_tree, Tree};
use crate::Result;

/// Gap handling options. Ambiguous means that gaps are treated as unknown characters (X),
/// Proper means that the gaps are treated as a separate character.
#[derive(Debug, Clone, PartialEq)]
pub enum GapHandling {
    Ambiguous,
    Proper,
    Undefined,
}

impl From<String> for GapHandling {
    fn from(gap_handling: String) -> GapHandling {
        GapHandling::from(gap_handling.as_str())
    }
}

impl From<&str> for GapHandling {
    fn from(gap_handling: &str) -> GapHandling {
        if gap_handling.to_lowercase().contains("ambig") {
            GapHandling::Ambiguous
        } else if gap_handling.to_lowercase().contains("proper") {
            GapHandling::Proper
        } else {
            GapHandling::Undefined
        }
    }
}

pub struct PhyloInfoBuilder {
    sequence_file: PathBuf,
    tree_file: Option<PathBuf>,
    gap_handling: GapHandling,
}

impl PhyloInfoBuilder {
    pub fn new(sequence_file: PathBuf) -> PhyloInfoBuilder {
        PhyloInfoBuilder {
            sequence_file,
            tree_file: None,
            gap_handling: GapHandling::Undefined,
        }
    }

    /// Creates a PhyloInfo struct from a two given files, one containing the sequences in fasta format and
    /// one containing the tree in newick format.
    /// The sequences might not be aligned.
    /// The ids of the tree leaves and provided sequences must match.
    /// In the output the sequences are sorted by the leaf ids and converted to uppercase.
    ///
    /// # Arguments
    /// * `sequence_file` - File path to the sequence fasta file.
    /// * `tree_file` - File path to the tree newick file.
    /// * `gap_handling` - Gap handling option -- treat gaps as ambiguous characters or as a separate character.
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
    /// assert!(info.has_msa());
    /// for (i, node) in info.tree.leaves().iter().enumerate() {
    ///     assert!(info.sequences[i].id() == node.id);
    /// }
    /// for rec in info.sequences.iter() {
    ///     assert!(!rec.seq().is_empty());
    ///     assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    /// }
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

    pub fn sequence_file(mut self, path: PathBuf) -> PhyloInfoBuilder {
        self.sequence_file = path;
        self
    }

    pub fn tree_file(mut self, path: PathBuf) -> PhyloInfoBuilder {
        self.tree_file = Some(path);
        self
    }

    pub fn gap_handling(mut self, gap_handling: GapHandling) -> PhyloInfoBuilder {
        self.gap_handling = gap_handling;
        self
    }

    pub(crate) fn build_from_objects(
        sequences: Vec<Record>,
        tree: Tree,
        gap_handling: GapHandling,
    ) -> Result<PhyloInfo> {
        Self::check_sequences_not_empty(&sequences)?;
        let sequences = make_sequences_uppercase(&sequences);

        Self::validate_tree_sequence_ids(&tree, &sequences)?;

        let msa: Option<Vec<Record>> = Self::msa_if_aligned(&sequences);
        let model_type = sequence_type(&sequences);
        Ok(PhyloInfo {
            sequences,
            model_type,
            tree,
            msa,
            gap_handling,
        })
    }

    pub fn build(self) -> Result<PhyloInfo> {
        info!(
            "Reading sequences from file {}",
            self.sequence_file.display()
        );
        let sequences = io::read_sequences_from_file(&self.sequence_file)?;
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
        Self::build_from_objects(sequences, tree, self.gap_handling)
    }

    /// Returns a vector of records representing the MSA if all the sequences are of the same length.
    /// Otherwise returns None.
    ///
    /// # Arguments
    /// * `sequences` - Vector of fasta records.
    fn msa_if_aligned(sequences: &[Record]) -> Option<Vec<Record>> {
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
    fn check_sequences_not_empty(sequences: &[Record]) -> Result<()> {
        if sequences.is_empty() {
            bail!(DataError {
                message: String::from("No sequences provided, aborting.")
            });
        }
        Ok(())
    }

    /// Checks that the ids of the tree leaves and the sequences match, bails with an error otherwise.
    fn validate_tree_sequence_ids(tree: &Tree, sequences: &[Record]) -> Result<()> {
        let tip_ids: HashSet<String> = HashSet::from_iter(tree.leaf_ids());
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

/// The PhyloInfo struct contains all the information needed to run a phylogenetic analysis.
///
/// # TODO:
/// * Add methods to protein alignments.
/// * Enure encoding matches model.
#[derive(Debug, Clone)]
pub struct PhyloInfo {
    /// Type of the sequences (DNA/Protein).
    pub model_type: ModelType,
    /// Unaligned sequences.
    pub sequences: Vec<Record>,
    /// Multiple sequence alignment of the sequences, if they are aligned.
    msa: Option<Vec<Record>>,
    /// Phylogenetic tree.
    pub tree: Tree,
    // pub leaf_encoding: HashMap<String, DMatrix<f64>>,
    gap_handling: GapHandling,
}

/// Converts the given sequences to uppercase and returns a new vector.
fn make_sequences_uppercase(sequences: &[Record]) -> Vec<Record> {
    sequences
        .iter()
        .map(|rec| Record::with_attrs(rec.id(), rec.desc(), &rec.seq().to_ascii_uppercase()))
        .collect()
}

impl PhyloInfo {
    pub fn aligned_sequence(&self, id: &str) -> Option<&Record> {
        self.msa
            .as_ref()
            .and_then(|msa| msa.iter().find(|rec| rec.id() == id))
    }

    pub fn sequence(&self, id: &str) -> Option<&Record> {
        self.sequences.iter().find(|rec| rec.id() == id)
    }

    pub fn has_msa(&self) -> bool {
        self.msa.is_some()
    }

    pub fn msa_length(&self) -> usize {
        self.msa.as_ref().map(|msa| msa[0].seq().len()).unwrap_or(0)
    }

    /// Returns the empirical frequencies of the symbols in the sequences.
    /// The frequencies are calculated from the unaligned sequences.
    /// If an unambiguous character is not present in the sequences, its frequency is artificially
    /// set to at least one.
    ///
    /// # Example
    /// ```
    /// use std::path::PathBuf;
    /// use phylo::frequencies;
    /// use phylo::phylo_info::{GapHandling, PhyloInfoBuilder};
    /// use phylo::substitution_models::FreqVector;
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA2_unaligned.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    ///     GapHandling::Proper)
    /// .build()
    /// .unwrap();
    /// let freqs = info.freqs();
    /// assert_eq!(freqs, frequencies!(&[1.0, 2.0, 5.0, 1.0]).scale(1.0 / 9.0));
    /// assert_eq!(freqs.sum(), 1.0);
    /// ```
    pub fn freqs(&self) -> FreqVector {
        let alphabet = alphabet_from_type(self.model_type, &self.gap_handling);
        let mut freqs = alphabet.empty_freqs();
        for &char in alphabet.symbols().iter().chain(alphabet.ambiguous()) {
            let count = self
                .sequences
                .iter()
                .map(|rec| rec.seq().iter().filter(|&c| c == &char).count())
                .sum::<usize>() as f64;
            freqs += alphabet.char_encoding(char).scale(count);
        }
        for char in alphabet.symbols().iter() {
            let idx = alphabet.index(char);
            if freqs[idx] == 0.0 {
                freqs[idx] = 1.0;
            }
        }
        freqs.scale_mut(1.0 / freqs.sum());
        freqs
    }

    /// Creates a vector of leaf encodings for the ungapped sequences.
    /// Used for the likelihood calculation to avoid having to get the character encoding
    /// from scratch every time the likelihood is optimised.
    pub fn leaf_encoding(&self) -> HashMap<String, DMatrix<f64>> {
        let alphabet = alphabet_from_type(self.model_type, &self.gap_handling);
        let mut leaf_encoding = HashMap::with_capacity(self.sequences.len());
        for seq in self.sequences.iter() {
            leaf_encoding.insert(
                seq.id().to_string(),
                DMatrix::from_columns(
                    seq.seq()
                        .iter()
                        .map(|&c| alphabet.char_encoding(c))
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
            );
        }
        leaf_encoding
    }
}

#[cfg(test)]
mod phylo_info_tests;
