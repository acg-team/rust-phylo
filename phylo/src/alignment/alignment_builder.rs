use std::collections::HashMap;

use anyhow::bail;

use crate::align;
use crate::alignment::{Alignment, InternalMapping, Mapping, PairwiseAlignment, Sequences};
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub struct AlignmentBuilder<'a> {
    tree: &'a Tree,
    seqs: Sequences,
    node_map: InternalMapping,
}

impl<'a> AlignmentBuilder<'a> {
    pub fn new(tree: &'a Tree, seqs: Sequences) -> AlignmentBuilder<'a> {
        AlignmentBuilder {
            tree,
            seqs,
            node_map: InternalMapping::new(),
        }
    }

    pub fn msa(mut self, msa: InternalMapping) -> Self {
        self.node_map = msa;
        self
    }

    pub fn build(self) -> Result<Alignment> {
        if !self.node_map.is_empty() {
            self.build_from_map()
        } else if self.seqs.aligned {
            self.build_from_seqs()
        } else {
            self.build_from_unaligned()
        }
    }

    /// This assumes that the tree structure matches the alignment structure and that the sequences are aligned.
    fn build_from_seqs(self) -> Result<Alignment> {
        let msa_len = self.seqs.msa_len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(self.tree.len());
        let mut msa = InternalMapping::with_capacity(self.tree.n);
        for node_idx in self.tree.postorder() {
            match node_idx {
                Int(_) => {
                    let childs = self.tree.children(node_idx);
                    let map_x = stack[&childs[0]].clone();
                    let map_y = stack[&childs[1]].clone();
                    stack.insert(*node_idx, Self::stack_maps(msa_len, &map_x, &map_y));
                    msa.insert(*node_idx, Self::clear_common_gaps(msa_len, &map_x, &map_y));
                }
                Leaf(_) => {
                    let seq = self.seqs.get_by_id(self.tree.node_id(node_idx)).seq();
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
        Ok(Alignment {
            seqs: self.seqs.without_gaps(),
            leaf_map: leaf_maps,
            node_map: msa,
        })
    }

    fn build_from_unaligned(self) -> Result<Alignment> {
        // TODO: use parsimony to align the sequences.
        bail!("Unaligned sequences are not yet supported.")
    }

    /// This assumes that the tree structure matches the alignment structure.
    fn build_from_map(self) -> Result<Alignment> {
        let mut alignment = Alignment {
            seqs: Sequences::new(Vec::new()),
            leaf_map: HashMap::new(),
            node_map: self.node_map,
        };
        let leaf_map = alignment.compile_leaf_map(&self.tree.root, self.tree)?;
        alignment.leaf_map = leaf_map;
        alignment.seqs = self.seqs.without_gaps();
        Ok(alignment)
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
