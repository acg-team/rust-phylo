use std::fmt::Display;

use anyhow::bail;
use bio::io::fasta::Record;
use hashbrown::HashMap;
use nalgebra::DMatrix;

use crate::alphabets::{Alphabet, GAP};
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::{align, Result};

pub mod sequences;
pub use sequences::*;
pub mod aligner;
pub use aligner::*;

pub type Position = Option<usize>;
pub type Mapping = Vec<Option<usize>>;
pub type InternalMapping = HashMap<NodeIdx, PairwiseAlignment>;
pub type LeafMapping = HashMap<NodeIdx, Mapping>;

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
    pub fn empty() -> PairwiseAlignment {
        PairwiseAlignment {
            map_x: Vec::new(),
            map_y: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Alignment {
    pub(crate) seqs: Sequences,
    pub(crate) leaf_map: LeafMapping,
    pub(crate) node_map: InternalMapping,
    /// Leaf sequence encodings.
    pub(crate) leaf_encoding: HashMap<String, DMatrix<f64>>,
}

impl Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.seqs)
    }
}

impl Alignment {
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
    pub fn alphabet(&self) -> &Alphabet {
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
    pub fn len(&self) -> usize {
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
    pub fn seq_count(&self) -> usize {
        self.leaf_map.len()
    }

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
        let mut msa = InternalMapping::with_capacity(tree.n);
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
            node_map: msa,
            leaf_encoding,
        })
    }

    pub fn compile(&self, tree: &Tree) -> Result<Sequences> {
        self.compile_subroot(None, tree)
    }

    pub(crate) fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_map.get(node).unwrap()
    }

    pub(crate) fn compile_subroot(
        &self,
        subroot_opt: Option<&NodeIdx>,
        tree: &Tree,
    ) -> Result<Sequences> {
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

        Ok(Sequences::with_alphabet(
            records,
            self.seqs.alphabet.clone(),
        ))
    }

    pub(crate) fn compile_leaf_map(&self, root: &NodeIdx, tree: &Tree) -> Result<LeafMapping> {
        let order = &tree.preorder_subroot(root);
        let msa_len = match root {
            Int(_) => self.node_map[root].map_x.len(),
            Leaf(_) => self.seqs.record_by_id(tree.node_id(root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        stack.insert(*root, (0..msa_len).map(Some).collect());
        let mut leaf_map = LeafMapping::with_capacity(tree.n);
        for idx in order {
            match idx {
                Int(_) => {
                    let parent = &stack[idx];
                    let childs = tree.children(idx);
                    let map_x = &self.node_map[idx].map_x;
                    let map_y = &self.node_map[idx].map_y;
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

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
