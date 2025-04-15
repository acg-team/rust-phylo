use std::collections::HashMap;
use std::fmt::Display;

use bio::io::fasta::Record;
use nalgebra::DMatrix;

use crate::alphabets::{Alphabet, GAP};
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub mod sequences;
pub use sequences::*;
pub mod alignment_builder;
pub use alignment_builder::*;

pub type Position = Option<usize>;
pub type Mapping = Vec<Option<usize>>;
pub type InternalMapping = HashMap<NodeIdx, PairwiseAlignment>;
pub type LeafMapping = HashMap<NodeIdx, Mapping>;
pub type SeqMapping = HashMap<NodeIdx, Mapping>;

#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseAlignment {
    map_x: Mapping,
    map_y: Mapping,
}

impl PairwiseAlignment {
    pub fn new(map_x: Mapping, map_y: Mapping) -> PairwiseAlignment {
        debug_assert!((map_x.len() == map_y.len()) | map_y.is_empty());
        PairwiseAlignment { map_x, map_y }
    }
}

#[derive(Debug, Clone)]
pub struct Alignment {
    pub(crate) seqs: Sequences,
    leaf_map: LeafMapping,
    node_map: InternalMapping,
    /// Leaf sequence encodings.
    pub(crate) leaf_encoding: HashMap<String, DMatrix<f64>>,
}

impl Display for Alignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.seqs)
    }
}

impl Alignment {
    pub fn alphabet(&self) -> &Alphabet {
        &self.seqs.alphabet
    }

    /// Returns the length of the MSA, i.e. the number of sites/columns.
    ///
    /// # Example
    /// ```
    /// use bio::io::fasta::Record;
    /// use phylo::tree::tree_parser::from_newick;
    /// use phylo::alignment::AlignmentBuilder;
    /// use phylo::alignment::sequences::Sequences;
    /// let tree = from_newick("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);")
    ///     .unwrap()
    ///     .pop()
    ///     .unwrap();
    /// let seqs = Sequences::new(vec![
    ///     Record::with_attrs("A0", Some("A0 sequence"), b"AAAA"),
    ///     Record::with_attrs("B1", Some("B1 sequence"), b"---A"),
    ///     Record::with_attrs("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = AlignmentBuilder::new(&tree, seqs).build().unwrap();
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
    /// use bio::io::fasta::Record;
    /// use phylo::tree::tree_parser::from_newick;
    /// use phylo::alignment::AlignmentBuilder;
    /// use phylo::alignment::sequences::Sequences;
    /// let tree = from_newick("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);")
    ///     .unwrap()
    ///     .pop()
    ///     .unwrap();
    /// let seqs = Sequences::new(vec![
    ///     Record::with_attrs("A0", Some("A0 sequence"), b"AAAA"),
    ///     Record::with_attrs("B1", Some("B1 sequence"), b"---A"),
    ///     Record::with_attrs("C2", Some("C2 sequence"), b"AA--"),
    /// ]);
    /// let msa = AlignmentBuilder::new(&tree, seqs).build().unwrap();
    /// assert_eq!(msa.seq_count(), 3);
    /// ```
    pub fn seq_count(&self) -> usize {
        self.leaf_map.len()
    }

    pub(crate) fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_map.get(node).unwrap()
    }

    pub(crate) fn compile(&self, subroot_opt: Option<&NodeIdx>, tree: &Tree) -> Result<Sequences> {
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

    fn compile_leaf_map(&self, root: &NodeIdx, tree: &Tree) -> Result<LeafMapping> {
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
}

#[derive(Debug, Clone)]
pub struct AncestralAlignment {
    pub(crate) seqs: Sequences,
    seq_map: SeqMapping,
    // TODO: implement  node_map: InternalMapping,
    #[allow(dead_code)]
    pub(crate) leaf_encoding: HashMap<String, DMatrix<f64>>,
}

impl AncestralAlignment {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.seq_map
            .values()
            .next()
            .map(|map| map.len())
            .unwrap_or(0)
    }
    pub fn seq_count(&self) -> usize {
        self.seq_map.len()
    }

    pub fn update_nodes(&mut self, new_nodes: SeqMapping) {
        for new_node in new_nodes.keys() {
            assert!(
                self.seq_map.contains_key(new_node),
                "The node that is to be updated ({new_node}) is not in the tree.",
            )
        }
        self.seq_map.extend(new_nodes);
    }
}

impl Display for AncestralAlignment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut internal = String::new();
        for (x, v) in self.seq_map.iter() {
            if !matches!(x, Int(_)) {
                continue;
            }
            internal.push_str(&format!("{x}, "));
            for opt in v {
                internal.push(if opt.is_some() { 'X' } else { '-' });
            }
            internal.push('\n');
        }
        write!(f, "{}{}", self.seqs, internal)
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
