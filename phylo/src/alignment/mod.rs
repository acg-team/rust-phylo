use std::collections::HashMap;

use bio::io::fasta::Record;

use crate::alphabets::GAP;
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub type Mapping = Vec<Option<usize>>;

pub struct AlignmentBuilder<'a> {
    tree: &'a Tree,
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
    sequences: &'a [Record],
    msa: Vec<Option<PairwiseAlignment>>,
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
    pub fn with_attrs(tree: &'a Tree) -> AlignmentBuilder<'a> {
        AlignmentBuilder { tree }
    }

    /// This assumes that the tree structure matches the alignment structure.
    pub fn build(self, sequences: &'a mut Vec<Record>) -> Result<Alignment<'a>> {
        debug_assert_eq!(sequences.len(), self.tree.n);
        debug_assert!(!sequences.is_empty());

        let mut alignment = vec![None; self.tree.nodes.len()];
        let msa_len = sequences[0].seq().len();
        for node in self.tree.postorder.iter() {
            match node {
                Int(idx) => {
                    let map: Mapping = (0..msa_len).map(Some).collect();
                    alignment[*idx] = Some(PairwiseAlignment {
                        map_x: map.clone(),
                        map_y: map,
                    });
                }
                Leaf(idx) => {
                    let rec = sequences
                        .iter_mut()
                        .find(|r| r.id() == self.tree.nodes[*idx].id)
                        .unwrap();

                    alignment[*idx] = Some(PairwiseAlignment {
                        map_x: align!(rec.seq()),
                        map_y: Vec::new(),
                    });
                    let mut sequence = rec.seq().to_vec();
                    sequence.retain(|c| c != &GAP);
                    *rec = Record::with_attrs(rec.id(), rec.desc(), &sequence);
                }
            }
        }
        Ok(Alignment {
            msa: alignment,
            tree: self.tree,
            sequences,
        })
    }
}

impl Alignment<'_> {
    fn sequence(&self, idx: usize) -> &Record {
        let id = &self.tree.nodes[idx].id;
        self.sequences.iter().find(|r| r.id() == id).unwrap()
    }

    pub fn compile(&self, subroot_idx: Option<NodeIdx>) -> Vec<Record> {
        let nodes = &self.tree.nodes;
        let order = self.tree.preorder_subroot(subroot_idx);

        let root_idx = usize::from(&order[0]);
        let msa_len = match &self.msa[usize::from(&order[0])] {
            Some(msa) => msa.map_x.len(),
            None => self.sequence(root_idx).seq().len(),
        };

        let mut stack = HashMap::<usize, Mapping>::with_capacity(nodes.len());
        stack.insert(root_idx, (0..msa_len).map(Some).collect());

        let mut msa = Vec::<Record>::with_capacity(self.tree.n);
        for node_idx in order {
            match node_idx {
                Int(idx) => {
                    let parent = stack[&idx].clone();
                    let children = self.msa[idx].as_ref().unwrap();
                    stack.insert(
                        usize::from(&nodes[idx].children[0]),
                        Self::map_node(&parent, &children.map_x),
                    );
                    stack.insert(
                        usize::from(&nodes[idx].children[1]),
                        Self::map_node(&parent, &children.map_y),
                    );
                }
                Leaf(idx) => {
                    let map = &self.msa[idx];
                    if let Some(map) = map {
                        if !map.map_x.is_empty() {
                            let parent = stack[&idx].clone();
                            let sequence_map = map.map_x.clone();
                            stack.remove(&idx);
                            stack.insert(idx, Self::map_node(&parent, &sequence_map));
                        }
                    }
                    let seq = self
                        .sequences
                        .iter()
                        .find(|r| r.id() == nodes[idx].id)
                        .unwrap();
                    let aligned_seq = Self::map_sequence(&stack[&idx], seq.seq());
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

    fn map_node(parent: &Mapping, child: &Mapping) -> Mapping {
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
