use crate::alignment::{
    AlignmentTrait, AncestralAlignment, AncestralAlignmentTrait, SeqMaps, Sequences,
};

use crate::asr::Asr;
use crate::tree::NodeIdx::{Internal, Leaf};
use crate::tree::{NodeIdx, Tree};
use crate::{record, Result};
use bio::io::fasta::Record;
use hashbrown::HashMap;

pub struct ParsimonyIndelSites {}

impl Asr for ParsimonyIndelSites {
    fn asr<L: AlignmentTrait>(
        &self,
        leaf_alignment: &L,
        tree: &Tree,
    ) -> Result<impl AncestralAlignmentTrait> {
        // the leaf_maps shouldn't contain any Internal keys, but filtering anyways
        let leaf_maps: SeqMaps = tree
            .iter()
            .filter(|node| matches!(node.idx, Leaf(_)))
            .map(|node| (node.idx, leaf_alignment.leaf_map(&node.idx).clone()))
            .collect();

        let (ancestral_maps, mut ancestral_seqs) =
            get_ancestral_maps_and_seqs(tree, leaf_alignment);
        let mut seqs = leaf_alignment.seqs().into_gapless();
        seqs.s.append(&mut ancestral_seqs);

        // not implemented for now and therefore just kept as empty HashMap
        let int_align_maps = HashMap::new();
        let idx_to_id = get_idx_to_id(tree, &seqs);
        let leaf_encoding = seqs.generate_leaf_encoding();

        Ok(AncestralAlignment::new(
            seqs,
            leaf_maps,
            ancestral_maps,
            int_align_maps,
            idx_to_id,
            leaf_encoding,
        ))
    }
}

/// Infers ancestral sequences for every site independently:
/// - insertion location is latest common ancestors of all non-gap characters
/// - deletion location is node n such that every leaf in the subtree rooted in n
///   is a gap and the parent of n is not a deletion location
fn get_ancestral_maps_and_seqs<L: AlignmentTrait>(
    tree: &Tree,
    leaf_alignment: &L,
) -> (SeqMaps, Vec<Record>) {
    let mut ancestral_maps: SeqMaps = HashMap::new();
    // counter[node] keeps track of the sequence index during the procedurally generated mapping for the node
    let mut counter = vec![0; tree.len()];
    for site in 0..leaf_alignment.len() {
        // upward pass
        let has_char = get_has_char(leaf_alignment, tree, site);
        // downward pass
        elongate_maps_and_seqs(tree, &mut ancestral_maps, &mut counter, site, has_char);
    }
    (ancestral_maps, ancestral_seqs(tree, counter))
}

fn elongate_maps_and_seqs(
    tree: &Tree,
    ancestral_maps: &mut HashMap<NodeIdx, Vec<Option<usize>>>,
    counter: &mut [usize],
    site: usize,
    has_char: Vec<bool>,
) {
    for node in tree.preorder() {
        if let Internal(_) = node {
            let children = &tree.node(node).children;
            let parent = &tree.node(node).parent;
            let both_have_chars =
                has_char[usize::from(children[0])] && has_char[usize::from(children[1])];
            let both_are_gap =
                !has_char[usize::from(children[0])] && !has_char[usize::from(children[1])];

            let char_was_chosen_for_parent = match parent {
                None => false,
                Some(parent) => ancestral_maps[parent][site].is_some(),
            };
            let must_choose_char = (both_have_chars || char_was_chosen_for_parent) && !both_are_gap;
            // appending to the mapping of the node
            match ancestral_maps.get_mut(node) {
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
                        ancestral_maps.insert(*node, vec![Some(0)]);
                        counter[usize::from(node)] = 1;
                    } else {
                        ancestral_maps.insert(*node, vec![None]);
                    }
                }
            };
        }
    }
}

fn get_has_char<L: AlignmentTrait>(leaf_alignment: &L, tree: &Tree, site: usize) -> Vec<bool> {
    // has_char[node] will be set to true if any leaf in the subtree rooted in node is not a gap
    let mut has_char = vec![false; tree.len()];
    for node_idx in tree.postorder() {
        match node_idx {
            Leaf(idx) => {
                // TODO: record_by_id is slow
                has_char[*idx] = leaf_alignment.leaf_map(node_idx)[site].is_some();
            }
            Internal(idx) => {
                let children = &tree.node(node_idx).children;
                has_char[*idx] =
                    has_char[usize::from(children[0])] || has_char[usize::from(children[1])];
            }
        }
    }
    has_char
}
fn get_idx_to_id(tree: &Tree, seqs: &Sequences) -> Vec<String> {
    let mut idx_to_id = vec![String::new(); seqs.len()];
    for node_idx in tree.postorder() {
        let record = seqs.record_by_id(tree.node_id(node_idx));
        idx_to_id[usize::from(node_idx)] = record.id().to_string();
    }
    idx_to_id
}

fn ancestral_seqs(tree: &Tree, counter: Vec<usize>) -> Vec<Record> {
    tree.postorder()
        .iter()
        .filter(|node_idx| matches!(node_idx, Internal(_)))
        .map(|node_idx| {
            record!(
                tree.node_id(node_idx),
                None,
                "X".repeat(counter[usize::from(node_idx)])
                    .to_string()
                    .as_bytes()
            )
        })
        .collect()
}

#[cfg(test)]
mod tests;
