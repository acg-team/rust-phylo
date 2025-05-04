use std::fmt::{Debug, Display};

use anyhow::bail;
use bio::io::fasta::Record;
use hashbrown::HashMap;
use nalgebra::DMatrix;

use crate::alphabets::{Alphabet, GAP};
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
pub trait AlignmentTrait: Display + Clone + Debug {
    fn alphabet(&self) -> &Alphabet;
    /// Returns the sequences without gaps
    fn seqs(&self) -> &Sequences;
    fn len(&self) -> usize;
    fn seq_count(&self) -> usize;
    fn leaf_map(&self, node: &NodeIdx) -> &Mapping;
    fn compile_subroot(&self, subroot: Option<&NodeIdx>, tree: &Tree) -> Result<Sequences>;
    fn leaf_encoding(&self) -> &HashMap<String, DMatrix<f64>>;
    fn int_align_map(&self) -> &InternalAlignments;
}

pub trait AncestralAlignmentTrait: AlignmentTrait {
    fn ancestral_map(&self, node: &NodeIdx) -> &Mapping;
}

#[derive(Debug, Clone)]
pub struct Alignment {
    // TODO: i could remove the pub crate here if i would write a ctor and getters, and maybe also adjust the compile_leaf_map()
    pub(crate) seqs: Sequences,
    pub(crate) leaf_map: SeqMaps,
    // TODO: alternative name: internal_alignments
    pub(crate) int_align_map: InternalAlignments,
    // then for the ancestral alignment i can do
    //  internal_maps: InternalMaps
    // if i implement getters for the maps and pass a NodeIdx to it then the implementations
    // for the AlignmentTrait::get_leaf_mapping and AncestralAlignmentTrait::get_ancestral_mapping have the same bodykdx
    /// Leaf sequence encodings.
    pub(crate) leaf_encoding: HashMap<String, DMatrix<f64>>,
}

impl Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.seqs)
    }
}

impl Alignment {
    /// Constructs an alignment instance from aligned sequences and a phylogenetic tree.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::alignment::Alignment;
    /// use phylo::alignment::Sequences;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = Alignment::from_aligned(seqs.clone(), &tree).unwrap();
    /// let aligned_seqs = msa.compile(&tree).unwrap();
    /// assert_eq!(aligned_seqs, seqs);
    /// ```
    pub fn from_aligned(mut seqs: Sequences, tree: &Tree) -> Result<Alignment> {
        if !seqs.aligned {
            bail!("Sequences are not aligned.")
        }
        seqs.remove_gap_cols();

        let msa_len = seqs.record(0).seq().len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        let mut msa = InternalAlignments::with_capacity(tree.n);
        for node_idx in tree.postorder() {
            match node_idx {
                Int(_) => {
                    let childs = tree.children(node_idx);
                    let map_x = stack[&childs[0]].clone();
                    let map_y = stack[&childs[1]].clone();
                    stack.insert(*node_idx, Self::stack_maps(msa_len, &map_x, &map_y));
                    msa.insert(*node_idx, Self::clear_common_gaps(msa_len, &map_x, &map_y));
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
        let leaf_encoding = seqs.generate_leaf_encoding();
        Ok(Alignment {
            seqs,
            leaf_map: leaf_maps,
            int_align_map: msa,
            leaf_encoding,
        })
    }

    pub fn compile(&self, tree: &Tree) -> Result<Sequences> {
        self.compile_subroot(None, tree)
    }

    pub(crate) fn compile_leaf_map(&self, root: &NodeIdx, tree: &Tree) -> Result<SeqMaps> {
        let order = &tree.preorder_subroot(root);
        let msa_len = match root {
            Int(_) => self.int_align_map[root].map_x.len(),
            Leaf(_) => self.seqs.record_by_id(tree.node_id(root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        stack.insert(*root, (0..msa_len).map(Some).collect());
        let mut leaf_map = SeqMaps::with_capacity(tree.n);
        for idx in order {
            match idx {
                Int(_) => {
                    let parent = &stack[idx];
                    let childs = tree.children(idx);
                    let map_x = &self.int_align_map[idx].map_x;
                    let map_y = &self.int_align_map[idx].map_y;
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

    fn map_sequence(map: &Mapping, seq: &[u8]) -> Vec<u8> {
        map.iter()
            .map(|site| if let Some(idx) = site { seq[*idx] } else { GAP })
            .collect()
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

impl AlignmentTrait for Alignment {
    fn int_align_map(&self) -> &InternalAlignments {
        &self.int_align_map
    }

    fn leaf_encoding(&self) -> &HashMap<String, DMatrix<f64>> {
        &self.leaf_encoding
    }
    fn seqs(&self) -> &Sequences {
        &self.seqs
    }

    /// Returns the alphabet of the MSA.
    ///
    /// # Example
    /// ```
    /// # use bio::io::fasta::Record;
    /// use phylo::alignment::Alignment;
    /// use phylo::alignment::Sequences;
    /// use phylo::alphabets::dna_alphabet;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::with_alphabet(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ], dna_alphabet());
    /// let msa = Alignment::from_aligned(seqs, &tree).unwrap();
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
    /// use phylo::alignment::Alignment;
    /// use phylo::alignment::Sequences;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = Alignment::from_aligned(seqs, &tree).unwrap();
    /// assert_eq!(msa.len(), 4);
    /// ```
    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> usize {
        self.leaf_map
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
    /// use phylo::alignment::Alignment;
    /// use phylo::alignment::Sequences;
    /// use phylo::{record, tree};
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = Alignment::from_aligned(seqs, &tree).unwrap();
    /// assert_eq!(msa.seq_count(), 3);
    /// ```
    fn seq_count(&self) -> usize {
        self.leaf_map.len()
    }

    fn compile_subroot(&self, subroot_opt: Option<&NodeIdx>, tree: &Tree) -> Result<Sequences> {
        let subroot = subroot_opt.unwrap_or(&tree.root);
        let map = if subroot == &tree.root {
            self.leaf_map.clone()
        } else {
            self.compile_leaf_map(subroot, tree)?
        };
        let mut records = Vec::with_capacity(map.len());
        for (idx, map) in &map {
            let rec = self.seqs.record_by_id(tree.node_id(idx));
            let aligned_seq = Self::map_sequence(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }

        Ok(Sequences::with_alphabet(records, self.seqs.alphabet))
    }

    fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_map.get(node).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct AncestralAlignment {
    /// This contains also the ancestral sequences
    seqs: Sequences,
    leaf_maps: SeqMaps,
    ancestral_maps: SeqMaps,
    int_align_maps: InternalAlignments,
    idx_to_id: Vec<String>,
    // TODO: maybe also set this to private and write getter with key = String
    pub(crate) leaf_encoding: HashMap<String, DMatrix<f64>>,
}

impl Display for AncestralAlignment {
    // TODO: this is the same as for the Alignment, but it does print the seqs
    //       which are unaligned
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let both_maps = self.leaf_maps.iter().chain(self.ancestral_maps.iter());
        for (node_idx, seq_map) in both_maps {
            let id = &self.idx_to_id[usize::from(node_idx)];
            let record = self.seqs.record_by_id(id);
            let aligned_seq = aligned_seq!(seq_map, record.seq());
            write!(f, "{}", record!(id, record.desc(), &aligned_seq))?;
        }
        Ok(())
    }
}

impl AncestralAlignment {
    pub fn new(
        seqs: Sequences,
        leaf_maps: SeqMaps,
        ancestral_maps: SeqMaps,
        int_align_maps: InternalAlignments,
        idx_to_id: Vec<String>,
        leaf_encoding: HashMap<String, DMatrix<f64>>,
    ) -> AncestralAlignment {
        AncestralAlignment {
            seqs,
            leaf_maps,
            ancestral_maps,
            idx_to_id,
            leaf_encoding,
            int_align_maps,
        }
    }

    /// the seqs should contain a seq for every node in the tree
    pub fn from_aligned(mut seqs: Sequences, tree: &Tree) -> Result<AncestralAlignment> {
        let mut leaf_maps = HashMap::<NodeIdx, Mapping>::new();
        let mut ancestral_maps = HashMap::<NodeIdx, Mapping>::new();
        seqs.remove_gap_cols();
        let mut idx_to_id = vec![String::new(); seqs.len()];
        for node_idx in tree.postorder() {
            let record = seqs.record_by_id(tree.node_id(node_idx));
            let mapping = align!(record.seq());
            match node_idx {
                Int(_) => ancestral_maps.insert(*node_idx, mapping),
                Leaf(_) => leaf_maps.insert(*node_idx, mapping),
            };
            idx_to_id[usize::from(node_idx)] = record.id().to_string();
        }
        let seqs = seqs.into_gapless();
        let leaf_encoding = seqs.generate_leaf_encoding();
        Ok(AncestralAlignment {
            seqs,
            leaf_maps,
            ancestral_maps,
            idx_to_id,
            leaf_encoding,
            int_align_maps: HashMap::<NodeIdx, PairwiseAlignment>::new(),
        })
    }
}

impl AlignmentTrait for AncestralAlignment {
    fn int_align_map(&self) -> &InternalAlignments {
        &self.int_align_maps
    }

    fn compile_subroot(&self, _: Option<&NodeIdx>, _: &Tree) -> Result<Sequences> {
        Ok(self.seqs.clone())
    }

    fn leaf_encoding(&self) -> &HashMap<String, DMatrix<f64>> {
        &self.leaf_encoding
    }
    fn seqs(&self) -> &Sequences {
        &self.seqs
    }

    fn alphabet(&self) -> &Alphabet {
        &self.seqs.alphabet
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
        self.leaf_maps.len() + self.ancestral_maps.len()
    }

    fn leaf_map(&self, node_idx: &NodeIdx) -> &Mapping {
        self.leaf_maps.get(node_idx).unwrap()
    }
}

impl AncestralAlignmentTrait for AncestralAlignment {
    fn ancestral_map(&self, node_idx: &NodeIdx) -> &Mapping {
        self.ancestral_maps.get(node_idx).unwrap()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
