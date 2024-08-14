use std::collections::HashMap;

use bio::io::fasta::Record;

use crate::alphabets::Alphabet;
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub mod sequences;
pub use sequences::*;
pub mod alignment_builder;
pub use alignment_builder::*;

#[macro_export]
macro_rules! align {
    ($e:expr) => {{
        use $crate::alphabets::GAP;
        let mut i = 0;
        $e.iter()
            .map(|&byte| {
                if byte == GAP {
                    None
                } else {
                    i += 1;
                    Some(i - 1)
                }
            })
            .collect::<Vec<_>>()
    }};
}

pub type Position = Option<usize>;
pub type Mapping = Vec<Option<usize>>;
pub type InternalMapping = HashMap<NodeIdx, PairwiseAlignment>;
pub type LeafMapping = HashMap<NodeIdx, Mapping>;

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
}

impl Alignment {
    pub fn alphabet(&self) -> &Alphabet {
        &self.seqs.alphabet
    }

    pub fn len(&self) -> usize {
        self.leaf_map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaf_map.is_empty()
    }

    pub fn msa_len(&self) -> usize {
        self.leaf_map
            .values()
            .next()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    pub fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_map.get(node).unwrap()
    }

    pub(crate) fn compile(&self, subroot_opt: Option<NodeIdx>, tree: &Tree) -> Result<Vec<Record>> {
        let subroot = subroot_opt.unwrap_or(tree.root);
        let map = if subroot == tree.root {
            self.leaf_map.clone()
        } else {
            self.compile_leaf_map(subroot, tree)?
        };
        let mut records = Vec::with_capacity(map.len());
        for (idx, map) in &map {
            let rec = self.seqs.get_by_id(tree.node_id(idx));
            let aligned_seq = Self::map_sequence(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }
        Ok(records)
    }

    fn compile_leaf_map(&self, root: NodeIdx, tree: &Tree) -> Result<LeafMapping> {
        let order = &tree.preorder_subroot(Some(root));
        let msa_len = match root {
            Int(_) => self.node_map[&root].map_x.len(),
            Leaf(_) => self.seqs.get_by_id(tree.node_id(&root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        stack.insert(root, (0..msa_len).map(Some).collect());
        let mut leaf_map = LeafMapping::with_capacity(tree.n);
        for idx in order {
            match idx {
                Int(_) => {
                    let parent = &stack[idx].clone();
                    let childs = tree.children(idx);
                    let map_x = &self.node_map[idx].map_x;
                    let map_y = &self.node_map[idx].map_y;
                    stack.insert(childs[0], Self::map_child(parent, map_x));
                    stack.insert(childs[1], Self::map_child(parent, map_y));
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
            .map(|site| {
                if let Some(idx) = site {
                    seq[*idx]
                } else {
                    b'-'
                }
            })
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

#[cfg(test)]
mod tests;
