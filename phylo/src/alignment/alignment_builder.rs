use std::collections::HashMap;

use anyhow::bail;
use bio::io::fasta::Record;

use crate::align;
use crate::alignment::{Alignment, InternalMapping, Mapping, PairwiseAlignment, Sequences};
use crate::alphabets::GAP;
use crate::parsimony::costs::{GapMultipliers, ParsimonyCostsSimple};
use crate::parsimony::pars_align_on_tree;
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub struct AlignmentBuilder<'a> {
    tree: &'a Tree,
    seqs: Sequences,
}

impl<'a> AlignmentBuilder<'a> {
    pub fn new(tree: &'a Tree, seqs: Sequences) -> AlignmentBuilder<'a> {
        AlignmentBuilder { tree, seqs }
    }

    pub fn build(self) -> Result<Alignment> {
        if self.seqs.aligned {
            self.reconstruct_from_aligned_seqs()
        } else {
            self.align_unaligned_seqs()
        }
    }

    fn align_unaligned_seqs(self) -> Result<Alignment> {
        let gap = GapMultipliers {
            open: 2.5,
            ext: 0.5,
        };
        let costs = ParsimonyCostsSimple::new(1.0, gap, self.seqs.alphabet());
        let (aligns, _scores) = pars_align_on_tree(&costs, self.tree, self.seqs.clone());
        let mut alignment = Alignment {
            seqs: Sequences::new(Vec::new()),
            leaf_map: HashMap::new(),
            node_map: aligns,
            leaf_encoding: HashMap::new(),
        };
        let leaf_map = alignment.compile_leaf_map(&self.tree.root, self.tree)?;
        alignment.leaf_map = leaf_map;
        alignment.seqs = self.seqs.into_gapless();
        alignment.leaf_encoding = alignment.seqs.generate_leaf_encoding();
        Ok(alignment)
    }

    /// This assumes that the tree structure matches the alignment structure and that the sequences are aligned.
    fn reconstruct_from_aligned_seqs(mut self) -> Result<Alignment> {
        if !self.seqs.aligned {
            bail!("Sequences are not aligned.")
        }

        self.remove_gap_cols();

        let msa_len = self.seqs.record(0).seq().len();
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
                    let seq = self.seqs.record_by_id(self.tree.node_id(node_idx)).seq();
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
        let seqs = self.seqs.into_gapless();
        let leaf_encoding = seqs.generate_leaf_encoding();
        Ok(Alignment {
            seqs,
            leaf_map: leaf_maps,
            node_map: msa,
            leaf_encoding,
        })
    }

    fn remove_gap_cols(&mut self) {
        let mut gap_cols = Vec::new();
        for col in 0..self.seqs.record(0).seq().len() {
            if self.seqs.iter().all(|rec| rec.seq()[col] == GAP) {
                gap_cols.push(col);
            }
        }
        let new_seqs = self.seqs.iter().map(|rec| {
            let seq: Vec<u8> = rec
                .seq()
                .iter()
                .enumerate()
                .filter(|(i, _)| !gap_cols.contains(i))
                .map(|(_, c)| *c)
                .collect();
            Record::with_attrs(rec.id(), rec.desc(), &seq)
        });
        self.seqs = Sequences::with_alphabet(new_seqs.collect(), self.seqs.alphabet().clone());
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
