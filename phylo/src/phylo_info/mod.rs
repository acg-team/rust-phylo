use std::collections::HashMap;

use anyhow::bail;
use bio::io::fasta::Record;
use nalgebra::DMatrix;

use crate::alignment::Alignment;
use crate::alphabets::alphabet_from_type;
use crate::evolutionary_models::ModelType;
use crate::substitution_models::FreqVector;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

mod phyloinfo_builder;
pub use phyloinfo_builder::*;

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

/// The PhyloInfo struct contains all the information needed to run a phylogenetic analysis.
///
/// # TODO:
/// * Add methods to protein alignments.
/// * Enure encoding matches model.
#[derive(Debug, Clone)]
pub struct PhyloInfo {
    /// Type of the sequences (DNA/Protein).
    pub model_type: ModelType,
    /// Multiple sequence alignment of the sequences
    pub msa: Alignment,
    /// Phylogenetic tree.
    pub tree: Tree,
    /// Leaf sequence encodings.
    pub leaf_encoding: HashMap<String, DMatrix<f64>>,
    /// Gap handling option.
    pub gap_handling: GapHandling,
}

impl PhyloInfo {
    pub fn msa_length(&self) -> usize {
        self.msa.msa_len()
    }

    pub fn compile_alignment(&self, subroot: Option<NodeIdx>) -> Result<Vec<Record>> {
        self.msa.compile(subroot, &self.tree)
    }

    pub fn leaf_encoding_by_id(&self, id: &str) -> Result<&DMatrix<f64>> {
        let encoding = self.leaf_encoding.get(id);
        if encoding.is_none() {
            bail!("No encoding found for leaf with id {}", id);
        }
        Ok(encoding.unwrap())
    }

    pub fn leaf_encoding(&self, idx: &NodeIdx) -> Result<&DMatrix<f64>> {
        let id = self.tree.node_id(idx);
        self.leaf_encoding_by_id(id)
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
    ///     PathBuf::from("./data/sequences_DNA1.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    ///     GapHandling::Proper)
    /// .build()
    /// .unwrap();
    /// let freqs = info.freqs();
    /// assert_eq!(freqs, frequencies!(&[1.25, 2.25, 4.25, 1.25]).scale(1.0 / 9.0));
    /// assert_eq!(freqs.sum(), 1.0);
    /// ```
    pub fn freqs(&self) -> FreqVector {
        let alphabet = alphabet_from_type(self.model_type, &self.gap_handling);
        let mut freqs = alphabet.empty_freqs();
        for &char in alphabet.symbols().iter().chain(alphabet.ambiguous()) {
            let count = self
                .msa
                .seqs
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
}

#[cfg(test)]
mod tests;
