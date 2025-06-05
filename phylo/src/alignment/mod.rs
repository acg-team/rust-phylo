use std::fmt::{Debug, Display};

use anyhow::bail;
use bio::io::fasta::Record;
use hashbrown::HashMap;

use crate::alphabets::Alphabet;
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_indel_points::ParsimonyIndelPoints;
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
    fn compile_subroot(&self, subroot: Option<&NodeIdx>, tree: &Tree) -> Result<Sequences>;
    fn internal_alignments(&self) -> &InternalAlignments;
    fn from_aligned(sequences: Sequences, tree: &Tree) -> Result<Self>;
    fn from_internal_alignments(
        seqs: &Sequences,
        internal_alignments: InternalAlignments,
        tree: &Tree,
    ) -> Self;
}

pub trait AncestralAlignment: Alignment {
    fn ancestral_seqs(&self) -> &Sequences;
    fn ancestral_map(&self, node_idx: &NodeIdx) -> &Mapping;
    fn from_aligned_with_ancestral(all_seqs: Sequences, tree: &Tree) -> Result<Self>;
}

#[derive(Debug, Clone)]
pub struct MSA {
    seqs: Sequences,
    leaf_maps: SeqMaps,
    internal_alignments: InternalAlignments,
}

impl Display for MSA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.seqs)
    }
}

impl MSA {
    pub fn compile(&self, tree: &Tree) -> Result<Sequences> {
        self.compile_subroot(None, tree)
    }

    pub(crate) fn compile_leaf_map(&self, root: &NodeIdx, tree: &Tree) -> Result<SeqMaps> {
        let msa_len = match root {
            Int(_) => self.internal_alignments[root].map_x.len(),
            Leaf(_) => self.seqs.record_by_id(tree.node_id(root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        stack.insert(*root, (0..msa_len).map(Some).collect());
        let mut leaf_map = SeqMaps::with_capacity(tree.n);
        for idx in &tree.preorder_subroot(root) {
            match idx {
                Int(_) => {
                    let parent = &stack[idx];
                    let childs = tree.children(idx);
                    let map_x = &self.internal_alignments[idx].map_x;
                    let map_y = &self.internal_alignments[idx].map_y;
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
        Ok(leaf_map)
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
    fn from_internal_alignments(
        seqs: &Sequences,
        internal_alignments: InternalAlignments,
        tree: &Tree,
    ) -> Self {
        let mut alignment = MSA {
            seqs: seqs.into_gapless(),
            leaf_maps: HashMap::new(),
            internal_alignments,
        };
        let leaf_map = alignment.compile_leaf_map(&tree.root, tree).unwrap();
        alignment.leaf_maps = leaf_map;
        alignment
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
    fn from_aligned(mut seqs: Sequences, tree: &Tree) -> Result<MSA> {
        if !seqs.aligned {
            bail!("Sequences are not aligned.")
        }
        if seqs.len() != tree.n {
            bail!("The number of provided seqs does not match the number of leafs in the tree")
        }
        seqs.remove_gap_cols();

        let msa_len = seqs.record(0).seq().len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        let mut internal_alignments = InternalAlignments::with_capacity(tree.n);
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
        Ok(MSA {
            seqs,
            leaf_maps,
            internal_alignments,
        })
    }

    fn internal_alignments(&self) -> &InternalAlignments {
        &self.internal_alignments
    }

    fn seqs(&self) -> &Sequences {
        &self.seqs
    }

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

    fn compile_subroot(&self, subroot_opt: Option<&NodeIdx>, tree: &Tree) -> Result<Sequences> {
        let subroot = subroot_opt.unwrap_or(&tree.root);
        let map = if subroot == &tree.root {
            self.leaf_maps.clone()
        } else {
            self.compile_leaf_map(subroot, tree)?
        };
        let mut records = Vec::with_capacity(map.len());
        for (idx, map) in &map {
            let rec = self.seqs.record_by_id(tree.node_id(idx));
            let aligned_seq = aligned_seq!(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }

        Ok(Sequences::with_alphabet(records, self.seqs.alphabet))
    }

    fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_maps.get(node).unwrap()
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

impl MASA {
    pub fn new(
        leaf_seqs: Sequences,
        ancestral_seqs: Sequences,
        leaf_maps: SeqMaps,
        ancestral_maps: SeqMaps,
        internal_alignments: InternalAlignments,
        idx_to_id: Vec<String>,
    ) -> MASA {
        MASA {
            leaf_seqs,
            ancestral_seqs,
            leaf_maps,
            ancestral_maps,
            internal_alignments,
            idx_to_id,
        }
    }
}

impl Alignment for MASA {
    fn from_internal_alignments(
        seqs: &Sequences,
        internal_alignments: InternalAlignments,
        tree: &Tree,
    ) -> Self {
        let mut alignment = MSA {
            seqs: seqs.into_gapless(),
            leaf_maps: HashMap::new(),
            internal_alignments,
        };
        let leaf_map = alignment.compile_leaf_map(&tree.root, tree).unwrap();
        alignment.leaf_maps = leaf_map;
        let asr = ParsimonyIndelPoints {};
        asr.reconstruct_ancestral_seqs(&alignment, tree).unwrap()
    }

    fn internal_alignments(&self) -> &InternalAlignments {
        &self.internal_alignments
    }

    // TODO: or do i want to include the ancestral seqs here?
    fn compile_subroot(&self, subroot_opt: Option<&NodeIdx>, tree: &Tree) -> Result<Sequences> {
        let subroot = subroot_opt.unwrap_or(&tree.root);
        let map = if subroot == &tree.root {
            self.leaf_maps.clone()
        } else {
            tree.preorder_subroot(subroot)
                .iter()
                .filter(|node_idx| matches!(node_idx, Leaf(_)))
                .map(|node_idx| (*node_idx, self.leaf_map(node_idx).clone()))
                .collect()
        };
        let mut records = Vec::with_capacity(map.len());
        for (idx, map) in &map {
            let rec = self.leaf_seqs.record_by_id(tree.node_id(idx));
            let aligned_seq = aligned_seq!(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }
        Ok(Sequences::with_alphabet(records, self.leaf_seqs.alphabet))
    }

    fn seqs(&self) -> &Sequences {
        &self.leaf_seqs
    }

    fn alphabet(&self) -> &Alphabet {
        &self.leaf_seqs.alphabet
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

    /// Assumes that nodes of the tree have unique ids and that the leaf ids match the ids of the
    /// sequences.
    fn from_aligned(sequences: Sequences, tree: &Tree) -> Result<Self> {
        let msa = MSA::from_aligned(sequences, tree)?;
        // TODO: do the internal alignments, build in the above line, conform with adding ancestral seqs?
        //       see also from_aligned_with_ancestral
        let asr = ParsimonyIndelPoints {};
        asr.reconstruct_ancestral_seqs(&msa, tree)
    }
}

impl AncestralAlignment for MASA {
    fn ancestral_map(&self, node: &NodeIdx) -> &Mapping {
        self.ancestral_maps.get(node).unwrap()
    }

    fn ancestral_seqs(&self) -> &Sequences {
        &self.ancestral_seqs
    }

    /// Assumes that nodes of the tree have unique ids and that they match the ids of the
    /// sequences.
    fn from_aligned_with_ancestral(mut all_seqs: Sequences, tree: &Tree) -> Result<MASA> {
        if !all_seqs.aligned {
            bail!("Sequences are not aligned.")
        }
        if all_seqs.len() != tree.len() {
            bail!("The number of seqs does not match the number of nodes in the tree")
        }

        all_seqs.remove_gap_cols();
        let mut leaf_maps = HashMap::<NodeIdx, Mapping>::new();
        let mut ancestral_maps = HashMap::<NodeIdx, Mapping>::new();
        let mut leaf_records = Vec::new();
        let mut ancestral_records = Vec::new();
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
        Ok(MASA {
            leaf_seqs,
            ancestral_seqs,
            leaf_maps,
            ancestral_maps,
            idx_to_id,
            internal_alignments: HashMap::<NodeIdx, PairwiseAlignment>::new(),
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
