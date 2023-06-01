use crate::{
    parsimony_alignment::sequence_idx,
    tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree},
};
use bio::io::fasta::Record;

pub(crate) type Mapping = Vec<Option<usize>>;

#[derive(Clone, Debug)]
pub(crate) struct Alignment {
    pub(crate) map_x: Mapping,
    pub(crate) map_y: Mapping,
}

impl Alignment {
    pub(super) fn new(x: Mapping, y: Mapping) -> Alignment {
        Alignment { map_x: x, map_y: y }
    }

    pub(super) fn empty() -> Alignment {
        Alignment {
            map_x: vec![],
            map_y: vec![],
        }
    }
}

pub(crate) fn compile_alignment_representation(
    tree: &Tree,
    sequences: &[Record],
    alignment: &[Alignment],
    subroot: Option<NodeIdx>,
) -> Vec<Record> {
    let subroot_idx = match subroot {
        Some(idx) => idx,
        None => tree.root,
    };
    let order = tree.preorder_subroot(subroot_idx);
    let mut alignment_stack =
        vec![Vec::<Option<usize>>::new(); tree.internals.len() + tree.leaves.len()];

    match subroot_idx {
        Int(idx) => alignment_stack[idx] = (0..alignment[idx].map_x.len()).map(Some).collect(),
        Leaf(idx) => return vec![sequences[idx].clone()],
    }

    let mut msa = Vec::<Record>::with_capacity(tree.leaves.len());
    for node_idx in order {
        match node_idx {
            Int(idx) => {
                let mut padded_map_x = vec![None; alignment_stack[idx].len()];
                let mut padded_map_y = vec![None; alignment_stack[idx].len()];
                for (mapping_index, site) in alignment_stack[idx].iter().enumerate() {
                    if let Some(index) = site {
                        padded_map_x[mapping_index] = alignment[idx].map_x[*index];
                        padded_map_y[mapping_index] = alignment[idx].map_y[*index];
                    }
                }
                match tree.internals[idx].children[0] {
                    Int(child_idx) => alignment_stack[child_idx] = padded_map_x,
                    Leaf(child_idx) => {
                        alignment_stack[tree.internals.len() + child_idx] = padded_map_x
                    }
                }
                match tree.internals[idx].children[1] {
                    Int(child_idx) => alignment_stack[child_idx] = padded_map_y,
                    Leaf(child_idx) => {
                        alignment_stack[tree.internals.len() + child_idx] = padded_map_y
                    }
                }
            }
            Leaf(idx) => {
                let mut sequence = vec![b'-'; alignment_stack[tree.internals.len() + idx].len()];
                for (alignment_index, site) in alignment_stack[tree.internals.len() + idx]
                    .iter()
                    .enumerate()
                {
                    if let Some(index) = site {
                        sequence[alignment_index] = sequences[idx].seq()[*index]
                    }
                }
                msa.push(Record::with_attrs(
                    sequences[idx].id(),
                    sequences[idx].desc(),
                    &sequence,
                ));
            }
        }
    }
    msa.sort_by(|a, b| sequence_idx(sequences, a).cmp(&sequence_idx(sequences, b)));
    msa
}
