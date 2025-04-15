use std::collections::HashMap;

use anyhow::bail;
use bio::io::fasta::Record;
use log::warn;

use crate::align;
use crate::alignment::{
    Alignment, AncestralAlignment, InternalMapping, Mapping, PairwiseAlignment, SeqMapping,
    Sequences,
};
use crate::alphabets::GAP;
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
        // TODO: use parsimony to align the sequences.
        warn!("Making an initial alignment using parsimony.");
        bail!("Alignment of unaligned sequences is not yet implemented.")
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
        self.seqs = Sequences::with_alphabet(new_seqs.collect(), *self.seqs.alphabet());
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

pub struct AncestralAlignmentBuilder<'a> {
    tree: &'a Tree,
    seqs: Sequences,
}

impl<'a> AncestralAlignmentBuilder<'a> {
    pub fn new(tree: &'a Tree, seqs: Sequences) -> AncestralAlignmentBuilder<'a> {
        AncestralAlignmentBuilder { tree, seqs }
    }

    fn build_from_only_aligned_leafs(self) -> std::result::Result<AncestralAlignment, String> {
        let leaf_maps: SeqMapping = self
            .tree
            .iter()
            .filter(|node| matches!(node.idx, NodeIdx::Leaf(_)))
            .map(|node| (node.idx, align!(self.seqs.record_by_id(&node.id).seq())))
            .collect();

        let alignment_len = self.seqs.s.first().map(|map| map.seq().len()).unwrap_or(0);
        let n_nodes = self.tree.len();

        // inferring ancestral sequences for every site independently:
        // insertion location is latest common ancestors of all non-gap characters
        // deletion location is node n such that every leaf in the subtree rooted in n
        // is a gap and the parent of n is not a deletion location
        let mut all_maps = leaf_maps;
        // counter[node] keeps track of the sequence index during the procedurally generated mapping for the node
        let mut counter = vec![0; n_nodes];
        for site in 0..alignment_len {
            // has_char[node] will be set to true if any leaf in the subtree rooted in node is not a gap
            let mut has_char = vec![false; n_nodes];
            // upward pass
            for node in self.tree.postorder() {
                match node {
                    Leaf(id) => {
                        // TODO: record_by_id is slow
                        has_char[*id] =
                            self.seqs.record_by_id(&self.tree.node(node).id).seq()[site] != GAP;
                    }
                    Int(id) => {
                        let children = &self.tree.node(node).children;
                        has_char[*id] = has_char[usize::from(children[0])]
                            || has_char[usize::from(children[1])];
                    }
                }
            }
            // downward pass
            for node in self.tree.preorder() {
                if let Int(_) = node {
                    let children = &self.tree.node(node).children;
                    let parent = &self.tree.node(node).parent;
                    let both_have_chars =
                        has_char[usize::from(children[0])] && has_char[usize::from(children[1])];
                    let both_are_gap =
                        !has_char[usize::from(children[0])] && !has_char[usize::from(children[1])];

                    let char_was_chosen_for_parent = match parent {
                        None => false,
                        Some(parent) => all_maps[parent][site].is_some(),
                    };
                    let must_choose_char =
                        (both_have_chars || char_was_chosen_for_parent) && !both_are_gap;
                    // appending to the mapping of the node
                    match all_maps.get_mut(node) {
                        Some(mapping) => {
                            if must_choose_char {
                                mapping.push(Some(counter[usize::from(node)]));
                                counter[usize::from(node)] += 1;
                            } else {
                                mapping.push(None);
                            }
                        }
                        None => {
                            if must_choose_char {
                                all_maps.insert(*node, vec![Some(0)]);
                                counter[usize::from(node)] = 1;
                            } else {
                                all_maps.insert(*node, vec![None]);
                            }
                        }
                    };
                }
            }
        }

        // TODO: these do not contain the ancestral wildcard seqs, do I want them included?
        let seqs = self.seqs.into_gapless();
        let leaf_encoding = seqs.generate_leaf_encoding();
        Ok(AncestralAlignment {
            seqs,
            seq_map: all_maps,
            leaf_encoding,
        })
    }

    fn build_from_aligned_seqs_with_ancestors(
        self,
    ) -> std::result::Result<AncestralAlignment, String> {
        let seq_map: SeqMapping = self
            .tree
            .iter()
            .map(|node| (node.idx, align!(self.seqs.record_by_id(&node.id).seq())))
            .collect();
        let seqs = self.seqs.into_gapless();
        let leaf_encoding = seqs.generate_leaf_encoding();
        Ok(AncestralAlignment {
            seqs,
            seq_map,
            leaf_encoding,
        })
    }

    pub fn build(self) -> std::result::Result<AncestralAlignment, String> {
        if self.seqs.aligned {
            if self.tree.len() == self.seqs.len() {
                self.build_from_aligned_seqs_with_ancestors()
            } else if self.tree.n == self.seqs.len() {
                self.build_from_only_aligned_leafs()
            } else {
                Err(
                    "The number of sequences does not match the number of nodes nor the number of leafs \
                    in the tree, which is required for an ancestral alignment."
                        .to_string(),
                )
            }
        } else {
            Err("Unaligned sequences are not yet supported.".to_string())
        }
    }
}
