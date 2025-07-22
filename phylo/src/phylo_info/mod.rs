use crate::alignment::{Alignment, InternalAlignments, Mapping, SeqMaps, Sequences};
use crate::substitution_models::FreqVector;
use crate::tree::NodeIdx::{Internal, Leaf};
use crate::tree::{NodeIdx, Tree};
use crate::{aligned_seq, Result};
use bio::io::fasta::Record;

mod phyloinfo_builder;
use hashbrown::HashMap;
pub use phyloinfo_builder::*;

/// The PhyloInfo struct contains all the information needed for phylogenetic inference.
///
/// The struct can be built using the PhyloInfoBuilder from at least a fasta sequence file.
/// At the moment the sequences need to be aligned.
/// The struct also holds the leaf sequence character encodings.
///
/// # TODO:
/// * Ensure encoding matches model.
/// * Add support for unaligned sequences.
#[derive(Debug, Clone)]
pub struct PhyloInfo<A: Alignment> {
    /// Multiple sequence alignment
    pub msa: A,
    /// Phylogenetic tree
    pub tree: Tree,
}

impl<A: Alignment> PhyloInfo<A> {
    /// Compiles a representation of the alignment in a vector of fasta records.
    /// The alignment is compiled from the subtree rooted at `subroot`.
    /// If `subroot` is None, the whole alignment is compiled.
    /// Bails if the tree does not contain the subroot or does not match the alignment.
    pub fn compile_alignment(&self, subroot_opt: Option<&NodeIdx>) -> Result<Sequences> {
        let subroot = subroot_opt.unwrap_or(&self.tree.root);
        let maps = if subroot == &self.tree.root {
            self.msa.leaf_maps().clone()
        } else {
            Self::compile_leaf_map(
                subroot,
                self.msa.internal_alignments(),
                self.msa.seqs(),
                &self.tree,
            )
        };
        let mut records = Vec::with_capacity(maps.len());
        for (idx, map) in &maps {
            let rec = self.msa.seqs().record_by_id(self.tree.node_id(idx));
            let aligned_seq = aligned_seq!(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }

        Ok(Sequences::with_alphabet(records, self.msa.seqs().alphabet))
    }

    pub(crate) fn compile_leaf_map(
        sub_root: &NodeIdx,
        internal_alignments: &InternalAlignments,
        seqs: &Sequences,
        tree: &Tree,
    ) -> SeqMaps {
        let msa_len = match sub_root {
            Internal(_) => internal_alignments[sub_root].map_x.len(),
            Leaf(_) => seqs.record_by_id(tree.node_id(sub_root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        stack.insert(*sub_root, (0..msa_len).map(Some).collect());
        let mut leaf_map = SeqMaps::with_capacity(tree.n);
        for idx in &tree.preorder_subroot(sub_root) {
            match idx {
                Internal(_) => {
                    let parent = &stack[idx];
                    let childs = tree.children(idx);
                    let map_x = &internal_alignments[idx].map_x;
                    let map_y = &internal_alignments[idx].map_y;
                    let x = Self::map_child(parent, map_x);
                    let y = Self::map_child(parent, map_y);
                    stack.insert(childs[0], x);
                    stack.insert(childs[1], y);
                }
                Leaf(_) => {
                    leaf_map.insert(*idx, stack[idx].clone());
                }
            }
        }
        leaf_map
    }

    fn map_child(parent: &Mapping, child: &Mapping) -> Mapping {
        parent
            .iter()
            .map(|site| {
                if let Some(idx) = site {
                    child[*idx]
                } else {
                    None
                }
            })
            .collect::<Mapping>()
    }

    /// Returns the empirical frequencies of the symbols in the sequences.
    /// The frequencies are calculated from the unaligned sequences.
    /// If an unambiguous character is not present in the sequences, its frequency is artificially
    /// set to at least one.
    ///
    /// # Example
    /// ```
    /// use phylo::frequencies;
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::substitution_models::FreqVector;
    /// let info = PhyloInfoBuilder::with_attrs(
    ///     "./examples/data/sequences_DNA1.fasta",
    ///     "./examples/data/tree_diff_branch_lengths_2.newick")
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
                .seqs()
                .iter()
                .map(|rec| rec.seq().iter().filter(|&c| c == &char).count())
                .sum::<usize>() as f64;
            let mut char_freq = alphabet.char_encoding(char).clone();
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
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
