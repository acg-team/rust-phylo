use std::collections::HashMap;

use anyhow::bail;
use bio::io::fasta::Record;

use crate::phylo_info::PhyloInfo;
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf};
use crate::Result;

pub type Mapping = Vec<Option<usize>>;

#[derive(Clone, Debug)]
pub struct Alignment {
    pub map_x: Mapping,
    pub map_y: Mapping,
}

impl Alignment {
    pub fn new(x: Mapping, y: Mapping) -> Alignment {
        Alignment { map_x: x, map_y: y }
    }

    pub fn empty() -> Alignment {
        Alignment {
            map_x: vec![],
            map_y: vec![],
        }
    }
}

pub(crate) fn sequence_idx(sequences: &[Record], search: &Record) -> usize {
    sequences
        .iter()
        .position(|r| r.id() == search.id())
        .unwrap()
}

pub fn compile_alignment_representation(
    info: &PhyloInfo,
    alignment: &HashMap<usize, Alignment>,
    subroot: Option<NodeIdx>,
) -> Result<Vec<Record>> {
    let tree = &info.tree;
    let sequences = &info.sequences;
    let subroot_idx = match subroot {
        Some(idx) => idx,
        None => tree.root,
    };
    let order = tree.preorder_subroot(subroot_idx);
    let mut alignment_stack = HashMap::<usize, Vec<Option<usize>>>::new();

    match subroot_idx {
        Int(idx) => {
            let align = alignment_at_int_node(alignment, idx)?;
            alignment_stack.insert(idx, (0..align.map_x.len()).map(Some).collect());
        }
        Leaf(idx) => return Ok(vec![sequences[idx].clone()]),
    }

    let mut msa = Vec::<Record>::with_capacity(tree.n);
    for node_idx in order {
        match node_idx {
            Int(idx) => {
                let mut padded_map_x = vec![None; alignment_stack[&idx].len()];
                let mut padded_map_y = vec![None; alignment_stack[&idx].len()];
                for (mapping_index, site) in alignment_stack[&idx].iter().enumerate() {
                    let align = alignment_at_int_node(alignment, idx)?;
                    if let Some(index) = site {
                        padded_map_x[mapping_index] = align.map_x[*index];
                        padded_map_y[mapping_index] = align.map_y[*index];
                    }
                }
                alignment_stack.insert(usize::from(&tree.nodes[idx].children[0]), padded_map_x);
                alignment_stack.insert(usize::from(&tree.nodes[idx].children[1]), padded_map_y);
            }
            Leaf(idx) => {
                let sequence = sequences
                    .iter()
                    .find(|r| r.id() == tree.nodes[idx].id)
                    .unwrap();
                let mut aligned_seq = vec![b'-'; alignment_stack[&idx].len()];
                for (alignment_index, site) in alignment_stack[&idx].iter().enumerate() {
                    if let Some(index) = site {
                        aligned_seq[alignment_index] = sequence.seq()[*index]
                    }
                }
                msa.push(Record::with_attrs(
                    sequence.id(),
                    sequence.desc(),
                    &aligned_seq,
                ));
            }
        }
    }
    msa.sort_by_key(|record| sequence_idx(sequences, record));
    Ok(msa)
}

fn alignment_at_int_node(alignment: &HashMap<usize, Alignment>, idx: usize) -> Result<&Alignment> {
    if let Some(align) = alignment.get(&idx) {
        Ok(align)
    } else {
        bail!("Alignment doesn't match tree structure.");
    }
}

#[cfg(test)]
mod alignment_tests;
