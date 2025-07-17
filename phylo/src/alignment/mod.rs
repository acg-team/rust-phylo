use std::fmt::{Debug, Display};

use anyhow::bail;
use hashbrown::HashMap;

use crate::alphabets::Alphabet;
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::{validate_ids_with_ancestors, validate_taxa_ids};
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::{align, aligned_seq, record, Result};

pub mod sequences;
pub use sequences::*;
pub mod aligner;
pub use aligner::*;

pub type Position = Option<usize>;
pub type Mapping = Vec<Option<usize>>;
pub type InternalAlignments = HashMap<NodeIdx, PairwiseAlignment>;
pub type SeqMaps = HashMap<NodeIdx, Mapping>;

#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseAlignment {
    pub map_x: Mapping,
    pub map_y: Mapping,
}

impl PairwiseAlignment {
    pub fn new(map_x: Mapping, map_y: Mapping) -> PairwiseAlignment {
        debug_assert!((map_x.len() == map_y.len()) | map_y.is_empty());
        PairwiseAlignment { map_x, map_y }
    }
}
#[allow(clippy::len_without_is_empty)]
pub trait Alignment: Display + Clone + Debug {
    fn alphabet(&self) -> &Alphabet;
    /// Returns the sequences without gaps
    fn seqs(&self) -> &Sequences;
    /// Returns the length of the sequences in the alignment
    fn len(&self) -> usize;
    fn seq_count(&self) -> usize;
    fn leaf_map(&self, node: &NodeIdx) -> &Mapping;
    fn leaf_maps(&self) -> &SeqMaps;
    fn internal_alignments(&self) -> &InternalAlignments;
    fn from_aligned(mut sequences: Sequences, tree: &Tree) -> Result<Self> {
        if !sequences.aligned {
            bail!("Sequences are not aligned.")
        }
        sequences.ids_are_unique()?;
        validate_taxa_ids(tree, &sequences)?;
        sequences.remove_gap_cols();
        Ok(Self::from_aligned_unchecked(sequences, tree))
    }
    fn from_aligned_unchecked(sequences: Sequences, tree: &Tree) -> Self;
}

pub trait AncestralAlignment: Alignment {
    fn ancestral_seqs(&self) -> &Sequences;
    fn ancestral_map(&self, node_idx: &NodeIdx) -> &Mapping;
    fn from_aligned_with_ancestral(mut all_seqs: Sequences, tree: &Tree) -> Result<Self> {
        if !all_seqs.aligned {
            bail!("Sequences are not aligned.")
        }
        all_seqs.ids_are_unique()?;
        validate_ids_with_ancestors(tree, &all_seqs)?;
        all_seqs.remove_gap_cols();
        Ok(Self::from_aligned_with_ancestral_unchecked(all_seqs, tree))
    }
    fn from_aligned_with_ancestral_unchecked(all_seqs: Sequences, tree: &Tree) -> Self;
}

#[derive(Debug, Clone)]
pub struct MSA {
    seqs: Sequences,
    leaf_maps: SeqMaps,
    internal_alignments: InternalAlignments,
    idx_to_id: Vec<String>,
}

impl Display for MSA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (node_idx, seq_map) in &self.leaf_maps {
            let id = &self.idx_to_id[usize::from(node_idx)];
            let record = self.seqs.record_by_id(id);
            let aligned_seq = aligned_seq!(seq_map, record.seq());
            write!(f, "{}", record!(id, record.desc(), &aligned_seq))?;
        }
        write!(f, "{}", self.seqs)
    }
}

impl MSA {
    fn stack_maps(msa_len: usize, map_x: &Mapping, map_y: &Mapping) -> Mapping {
        let mut map = Vec::with_capacity(msa_len);
        let mut ind = 0usize;
        for (x, y) in map_x.iter().zip(map_y.iter()) {
            if x.is_none() && y.is_none() {
                map.push(None);
            } else {
                map.push(Some(ind));
                ind += 1;
            }
        }
        map
    }

    fn clear_common_gaps(msa_len: usize, map_x: &Mapping, map_y: &Mapping) -> PairwiseAlignment {
        let mut upd_map_x = Vec::with_capacity(msa_len);
        let mut upd_map_y = Vec::with_capacity(msa_len);
        for (x, y) in map_x.iter().zip(map_y.iter()) {
            if x.is_some() || y.is_some() {
                upd_map_x.push(*x);
                upd_map_y.push(*y);
            }
        }
        PairwiseAlignment::new(upd_map_x, upd_map_y)
    }
}

impl Alignment for MSA {
    /// Returns the alphabet of the MSA.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::alignment::{MSA, Alignment};
    /// use phylo::alignment::Sequences;
    /// use phylo::alphabets::dna_alphabet;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::with_alphabet(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ], dna_alphabet());
    /// let msa = MSA::from_aligned(seqs, &tree).unwrap();
    /// assert_eq!(*msa.alphabet(), dna_alphabet());
    ///
    fn alphabet(&self) -> &Alphabet {
        &self.seqs.alphabet
    }

    fn seqs(&self) -> &Sequences {
        &self.seqs
    }

    /// Returns the length of the MSA, i.e. the number of sites/columns.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::alignment::{MSA, Alignment};
    /// use phylo::alignment::Sequences;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = MSA::from_aligned(seqs, &tree).unwrap();
    /// assert_eq!(msa.len(), 4);
    /// ```
    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> usize {
        self.leaf_maps
            .values()
            .next()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    /// Returns the number of sequences in the MSA, i.e. the number of rows.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::alignment::{MSA, Alignment};
    /// use phylo::alignment::Sequences;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = MSA::from_aligned(seqs, &tree).unwrap();
    /// assert_eq!(msa.seq_count(), 3);
    /// ```
    fn seq_count(&self) -> usize {
        self.leaf_maps.len()
    }

    fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_maps.get(node).unwrap()
    }

    fn leaf_maps(&self) -> &SeqMaps {
        &self.leaf_maps
    }

    fn internal_alignments(&self) -> &InternalAlignments {
        &self.internal_alignments
    }

    /// Constructs an alignment instance from aligned sequences and a phylogenetic tree.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::alignment::{MSA, Alignment};
    /// use phylo::alignment::Sequences;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = MSA::from_aligned(seqs.clone(), &tree).unwrap();
    /// let aligned_seqs = msa.compile(&tree).unwrap();
    /// assert_eq!(aligned_seqs, seqs);
    /// ```
    fn from_aligned_unchecked(seqs: Sequences, tree: &Tree) -> MSA {
        let msa_len = seqs.record(0).seq().len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        let mut internal_alignments = InternalAlignments::with_capacity(tree.n);
        let mut idx_to_id = vec![String::new(); tree.len()];
        for node_idx in tree.postorder() {
            match node_idx {
                Int(_) => {
                    let childs = tree.children(node_idx);
                    let map_x = stack[&childs[0]].clone();
                    let map_y = stack[&childs[1]].clone();
                    stack.insert(*node_idx, Self::stack_maps(msa_len, &map_x, &map_y));
                    internal_alignments
                        .insert(*node_idx, Self::clear_common_gaps(msa_len, &map_x, &map_y));
                }
                Leaf(_) => {
                    let seq = seqs.record_by_id(tree.node_id(node_idx)).seq();
                    stack.insert(*node_idx, align!(seq).clone());
                    idx_to_id[usize::from(node_idx)] = tree.node_id(node_idx).to_string();
                }
            }
        }
        let leaf_maps = stack
            .iter()
            .filter_map(|(idx, map)| match idx {
                Leaf(_) => Some((*idx, map.clone())),
                _ => None,
            })
            .collect();

        let seqs = seqs.into_gapless();
        MSA {
            seqs,
            leaf_maps,
            internal_alignments,
            idx_to_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MASA {
    leaf_seqs: Sequences,
    ancestral_seqs: Sequences,
    leaf_maps: SeqMaps,
    ancestral_maps: SeqMaps,
    // TODO: this needs to be implemented
    internal_alignments: InternalAlignments,
    idx_to_id: Vec<String>,
}

impl Display for MASA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let both_maps = self.leaf_maps.iter().chain(self.ancestral_maps.iter());
        for (node_idx, seq_map) in both_maps {
            let id = &self.idx_to_id[usize::from(node_idx)];
            let record = match node_idx {
                Int(_) => self.ancestral_seqs.record_by_id(id),
                Leaf(_) => self.leaf_seqs.record_by_id(id),
            };
            let aligned_seq = aligned_seq!(seq_map, record.seq());
            write!(f, "{}", record!(id, record.desc(), &aligned_seq))?;
        }
        Ok(())
    }
}

impl Alignment for MASA {
    fn alphabet(&self) -> &Alphabet {
        &self.leaf_seqs.alphabet
    }

    fn seqs(&self) -> &Sequences {
        &self.leaf_seqs
    }
    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> usize {
        self.leaf_maps
            .values()
            .next()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    fn seq_count(&self) -> usize {
        self.leaf_maps.len()
    }

    fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_maps.get(node).unwrap()
    }

    fn leaf_maps(&self) -> &SeqMaps {
        &self.leaf_maps
    }

    fn internal_alignments(&self) -> &InternalAlignments {
        &self.internal_alignments
    }

    fn from_aligned(sequences: Sequences, tree: &Tree) -> Result<Self> {
        let msa = MSA::from_aligned(sequences, tree)?;
        // TODO: do the internal alignments, build in the above line, conform with adding ancestral seqs?
        //       see also from_aligned_with_ancestral
        let asr = ParsimonyPresenceAbsence {};
        asr.reconstruct_ancestral_seqs(&msa, tree)
    }
    fn from_aligned_unchecked(sequences: Sequences, tree: &Tree) -> Self {
        let msa = MSA::from_aligned_unchecked(sequences, tree);
        // TODO: do the internal alignments, build in the above line, conform with adding ancestral seqs?
        //       see also from_aligned_with_ancestral
        let asr = ParsimonyPresenceAbsence {};
        asr.reconstruct_ancestral_seqs_unchecked(&msa, tree)
    }
}

impl AncestralAlignment for MASA {
    fn ancestral_map(&self, node: &NodeIdx) -> &Mapping {
        self.ancestral_maps.get(node).unwrap()
    }

    fn ancestral_seqs(&self) -> &Sequences {
        &self.ancestral_seqs
    }

    fn from_aligned_with_ancestral_unchecked(all_seqs: Sequences, tree: &Tree) -> MASA {
        let mut leaf_maps = HashMap::<NodeIdx, Mapping>::with_capacity(tree.n);
        let mut ancestral_maps = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len() - tree.n);
        let mut leaf_records = Vec::with_capacity(tree.n);
        let mut ancestral_records = Vec::with_capacity(tree.len() - tree.n);
        let mut idx_to_id = vec![String::new(); tree.len()];
        for node_idx in tree.postorder() {
            let record = all_seqs.record_by_id(tree.node_id(node_idx));
            let mapping = align!(record.seq());
            match node_idx {
                Int(_) => {
                    ancestral_maps.insert(*node_idx, mapping);
                    ancestral_records.push(record.clone());
                }
                Leaf(_) => {
                    leaf_maps.insert(*node_idx, mapping);
                    leaf_records.push(record.clone());
                }
            };
            idx_to_id[usize::from(node_idx)] = record.id().to_string();
        }
        let leaf_seqs = Sequences {
            s: leaf_records,
            aligned: true,
            alphabet: all_seqs.alphabet,
        };
        let leaf_seqs = leaf_seqs.into_gapless();
        let ancestral_seqs = Sequences {
            s: ancestral_records,
            aligned: true,
            alphabet: all_seqs.alphabet,
        };
        let ancestral_seqs = ancestral_seqs.into_gapless();

        // TODO: internal_alignments still missing. How do they work if there are seqs at internal nodes?
        //       see also MASA::from_aligned
        MASA {
            leaf_seqs,
            ancestral_seqs,
            leaf_maps,
            ancestral_maps,
            idx_to_id,
            internal_alignments: HashMap::<NodeIdx, PairwiseAlignment>::new(),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
