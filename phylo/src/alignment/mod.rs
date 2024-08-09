use std::collections::HashMap;

use bio::io::fasta::Record;

use crate::alphabets::GAP;
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub type Position = Option<usize>;
pub type Mapping = Vec<Option<usize>>;

pub struct AlignmentBuilder<'a> {
    tree: &'a Tree,
    aligned_seqs: &'a [Record],
}

#[derive(Clone, Debug)]
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
    sequences: Vec<Record>,
    leaf_maps: Vec<Mapping>,
    msa: HashMap<NodeIdx, PairwiseAlignment>,
}

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

impl<'a> AlignmentBuilder<'a> {
    pub fn with_attrs(tree: &'a Tree, aligned_seqs: &'a [Record]) -> AlignmentBuilder<'a> {
        AlignmentBuilder { tree, aligned_seqs }
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

    /// This assumes that the tree structure matches the alignment structure.
    pub fn from_aligned_sequences(self) -> Result<Alignment<'a>> {
        debug_assert_eq!(self.aligned_seqs.len(), self.tree.n);
        debug_assert!(!self.aligned_seqs.is_empty());
        let msa_len = self.aligned_seqs[0].seq().len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(self.tree.nodes.len());
        let mut msa = HashMap::<NodeIdx, PairwiseAlignment>::with_capacity(self.tree.n);
        for node in self.tree.postorder.iter() {
            match node {
                Int(_) => {
                    let childs = self.tree.children(node);
                    let map_x = stack[&childs[0]].clone();
                    let map_y = stack[&childs[1]].clone();
                    stack.insert(*node, Self::stack_maps(msa_len, &map_x, &map_y));
                    msa.insert(*node, Self::clear_common_gaps(msa_len, &map_x, &map_y));
                }
                Leaf(idx) => {
                    let seq = self
                        .aligned_seqs
                        .iter()
                        .find(|r| r.id() == self.tree.nodes[*idx].id)
                        .unwrap()
                        .seq();
                    debug_assert!(seq.len() == msa_len);
                    stack.insert(*node, align!(seq).clone());
                }
            }
        }
        let sequences = Self::remove_gaps(self.aligned_seqs);
        let leaf_maps = sequences.iter().map(|rec| align!(rec.seq())).collect();
        Ok(Alignment {
            tree: self.tree,
            sequences,
            leaf_maps,
            msa,
        })
    }

    fn remove_gaps(aligned_seqs: &[Record]) -> Vec<Record> {
        aligned_seqs
            .iter()
            .map(|rec| {
                let sequence = rec
                    .seq()
                    .iter()
                    .filter(|&c| c != &GAP)
                    .copied()
                    .collect::<Vec<u8>>();
                Record::with_attrs(rec.id(), rec.desc(), &sequence)
            })
            .collect()
    }
}

impl Alignment<'_> {
    fn sequence(&self, idx: NodeIdx) -> &Record {
        let id = &self.tree.nodes[usize::from(idx)].id;
        self.sequences.iter().find(|r| r.id() == id).unwrap()
    }

    pub fn compile(&self, subroot_idx: Option<NodeIdx>) -> Vec<Record> {
        let nodes = &self.tree.nodes;
        let order = self.tree.preorder_subroot(subroot_idx);

        let root = order[0];
        let msa_len = match root {
            Int(_) => self.msa[&root].map_x.len(),
            Leaf(_) => self.sequence(root).seq().len(),
        };

        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(nodes.len());
        stack.insert(root, (0..msa_len).map(Some).collect());

        let mut msa = Vec::<Record>::with_capacity(self.tree.n);
        for node in &order {
            match node {
                Int(idx) => {
                    let parent = &stack[node].clone();
                    let map_x = &self.msa[node].map_x;
                    let map_y = &self.msa[node].map_y;
                    stack.insert(nodes[*idx].children[0], Self::map_child(parent, map_x));
                    stack.insert(nodes[*idx].children[1], Self::map_child(parent, map_y));
                }
                Leaf(idx) => {
                    let seq = self
                        .sequences
                        .iter()
                        .find(|r| r.id() == nodes[*idx].id)
                        .unwrap();
                    let aligned_seq = Self::map_sequence(&stack[node], seq.seq());
                    msa.push(Record::with_attrs(seq.id(), seq.desc(), &aligned_seq));
                }
            }
        }
        msa
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
mod alignment_tests;
