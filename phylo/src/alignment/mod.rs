use std::collections::HashMap;

use bio::io::fasta::Record;

use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};

pub mod sequences;
pub use sequences::*;

pub type Position = Option<usize>;
pub type Mapping = Vec<Option<usize>>;
pub type NodeMapping = HashMap<NodeIdx, PairwiseAlignment>;
pub type LeafMapping = HashMap<NodeIdx, Mapping>;

macro_rules! align {
    ($e:expr) => {{
        let mut i = 0;
        $e.iter()
            .map(|&byte| {
                if byte == b'-' {
                    None
                } else {
                    i += 1;
                    Some(i - 1)
                }
            })
            .collect::<Vec<_>>()
    }};
}

pub struct AlignmentBuilder<'a> {
    tree: &'a Tree,
    seqs: &'a Sequences,
    node_map: NodeMapping,
}

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
pub struct Alignment<'a> {
    tree: &'a Tree,
    seqs: Sequences,
    leaf_map: LeafMapping,
    node_map: NodeMapping,
}

/// TODO: Implement try_build.
impl<'a> AlignmentBuilder<'a> {
    pub fn new(tree: &'a Tree, seqs: &'a Sequences) -> AlignmentBuilder<'a> {
        AlignmentBuilder {
            tree,
            seqs,
            node_map: HashMap::new(),
        }
    }

    pub fn msa(mut self, msa: NodeMapping) -> Self {
        self.node_map = msa;
        self
    }

    pub fn build(self) -> Alignment<'a> {
        if self.node_map.is_empty() {
            println!("Building from sequences");
            self.build_from_seqs()
        } else {
            self.build_from_map()
        }
    }

    /// This assumes that the tree structure matches the alignment structure and that the sequences are aligned.
    fn build_from_seqs(self) -> Alignment<'a> {
        let msa_len = self.seqs.msa_len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(self.tree.len());
        let mut msa = NodeMapping::with_capacity(self.tree.n);
        for node in self.tree.postorder.iter() {
            match node {
                Int(_) => {
                    let childs = self.tree.children(node);
                    let map_x = stack[&childs[0]].clone();
                    let map_y = stack[&childs[1]].clone();
                    stack.insert(*node, Self::stack_maps(msa_len, &map_x, &map_y));
                    msa.insert(*node, Self::clear_common_gaps(msa_len, &map_x, &map_y));
                }
                Leaf(_) => {
                    let seq = self.seqs.get_by_id(self.tree.node_id(node)).seq();
                    stack.insert(*node, align!(seq).clone());
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
        Alignment {
            tree: self.tree,
            seqs: self.seqs.without_gaps(),
            leaf_map: leaf_maps,
            node_map: msa,
        }
    }

    /// This assumes that the tree structure matches the alignment structure.
    fn build_from_map(self) -> Alignment<'a> {
        let mut alignment = Alignment {
            tree: self.tree,
            seqs: Sequences::new(Vec::new()),
            leaf_map: HashMap::new(),
            node_map: self.node_map,
        };
        let leaf_map = alignment.compile_leaf_map(self.tree.root);
        alignment.leaf_map = leaf_map;
        alignment.seqs = self.seqs.without_gaps();
        alignment
    }

    fn stack_maps(msa_len: usize, map_x: &Mapping, map_y: &Mapping) -> Mapping {
        let mut map = Vec::with_capacity(msa_len);
        let mut ind: usize = 0;
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
            if x.is_none() && y.is_none() {
                continue;
            } else {
                upd_map_x.push(*x);
                upd_map_y.push(*y);
            }
        }
        PairwiseAlignment::new(upd_map_x, upd_map_y)
    }
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

impl Alignment<'_> {
    pub fn compile(&self, subroot_opt: Option<NodeIdx>) -> Vec<Record> {
        let subroot = subroot_opt.unwrap_or(self.tree.root);
        let map = if subroot == self.tree.root {
            self.leaf_map.clone()
        } else {
            self.compile_leaf_map(subroot)
        };
        let mut records = Vec::with_capacity(map.len());
        for (idx, map) in &map {
            let rec = self.seqs.get_by_id(self.tree.node_id(idx));
            let aligned_seq = map_sequence(map, rec.seq());
            records.push(Record::with_attrs(rec.id(), rec.desc(), &aligned_seq));
        }
        records
    }

    fn compile_leaf_map(&self, root: NodeIdx) -> LeafMapping {
        let order = &self.tree.preorder_subroot(Some(root));
        let msa_len = match root {
            Int(_) => self.node_map[&root].map_x.len(),
            Leaf(_) => self.seqs.get_by_id(self.tree.node_id(&root)).seq().len(),
        };
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(self.tree.len());
        stack.insert(root, (0..msa_len).map(Some).collect());
        let mut leaf_map = LeafMapping::with_capacity(self.tree.n);
        for idx in order {
            match idx {
                Int(_) => {
                    let parent = &stack[idx].clone();
                    let childs = self.tree.children(idx);
                    let map_x = &self.node_map[idx].map_x;
                    let map_y = &self.node_map[idx].map_y;
                    stack.insert(childs[0], map_child(parent, map_x));
                    stack.insert(childs[1], map_child(parent, map_y));
                }
                Leaf(_) => {
                    leaf_map.insert(*idx, stack[idx].clone());
                }
            }
        }
        leaf_map
    }
}

#[cfg(test)]
mod alignment_tests;
