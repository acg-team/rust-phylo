use std::collections::HashMap;

use anyhow::bail;
use bio::io::fasta::Record;
use nalgebra::DMatrix;

use crate::alignment::Alignment;
use crate::substitution_models::FreqVector;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

mod phyloinfo_builder;
pub use phyloinfo_builder::*;

/// The PhyloInfo struct contains all the information needed for phylogenetic inference.
///
/// The struct can be built using the PhyloInfoBuilder from at least a fasta sequnce file.
/// At the moment the sequences need to be aligned.
/// The struct also holds the leaf sequence character encodings.
///
/// # TODO:
/// * Enure encoding matches model.
/// * Add support for unaligned sequences.
#[derive(Debug, Clone)]
pub struct PhyloInfo {
    /// Multiple sequence alignment of the sequences
    pub msa: Alignment,
    /// Phylogenetic tree.
    pub tree: Tree,
    /// Leaf sequence encodings.
    pub leaf_encoding: HashMap<String, DMatrix<f64>>,
}

impl PhyloInfo {
    /// Returns the number of sites in the alignment.
    pub fn msa_length(&self) -> usize {
        self.msa.msa_len()
    }

    /// Compiles a represenataion of the alignment in a vector of fasta records.
    /// The alignment is compiled from the subtree rooted at `subroot`.
    /// If `subroot` is None, the whole alignment is compiled.
    /// Bails if the tree does not contain the subroot or does not match the alignment.
    pub fn compile_alignment(&self, subroot: Option<&NodeIdx>) -> Result<Vec<Record>> {
        self.msa.compile(subroot, &self.tree)
    }

    /// Returns the encoding of a leaf sequence by its id.
    pub fn leaf_encoding_by_id(&self, id: &str) -> Result<&DMatrix<f64>> {
        let encoding = self.leaf_encoding.get(id);
        if encoding.is_none() {
            bail!("No encoding found for leaf with id {}", id);
        }
        Ok(encoding.unwrap())
    }

    /// Returns the encoding of a leaf sequence by the leaf node index.
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
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::substitution_models::FreqVector;
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     PathBuf::from("./data/sequences_DNA1.fasta"),
    ///     PathBuf::from("./data/tree_diff_branch_lengths_2.newick"))
    /// .build()
    /// .unwrap();
    /// let freqs = info.freqs();
    /// assert_eq!(freqs, frequencies!(&[1.25, 2.25, 4.25, 1.25]).scale(1.0 / 9.0));
    /// assert_eq!(freqs.sum(), 1.0);
    /// ```
    pub fn freqs(&self) -> FreqVector {
        let alphabet = self.msa.alphabet();
        let mut freqs = alphabet.empty_freqs();
        for &char in alphabet.symbols().iter().chain(alphabet.ambiguous()) {
            let count = self
                .msa
                .seqs
                .iter()
                .map(|rec| rec.seq().iter().filter(|&c| c == &char).count())
                .sum::<usize>() as f64;
            let mut char_freq = alphabet.char_encoding(char);
            char_freq.scale_mut(1.0 / char_freq.sum());
            freqs += char_freq.scale(count);
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

    /// Creates a the character encoding for each given ungapped sequence.
    /// Used for the likelihood calculation to avoid having to get the character encoding
    /// from scratch every time the likelihood is optimised.
    fn generate_leaf_encoding(&mut self) {
        let alphabet = self.msa.alphabet();
        let mut leaf_encoding = HashMap::with_capacity(self.msa.len());
        for seq in self.msa.seqs.iter() {
            if seq.seq().is_empty() {
                leaf_encoding.insert(seq.id().to_string(), DMatrix::zeros(0, 0));
                continue;
            }
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
        self.leaf_encoding = leaf_encoding;
    }
}

#[cfg(test)]
mod tests;
