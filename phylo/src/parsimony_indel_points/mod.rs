use crate::alignment::{Alignment, AncestralAlignment, Sequences};

use crate::alphabets::GAP;
use crate::asr::AncestralSequenceReconstruction;
use crate::tree::NodeIdx::{Internal, Leaf};
use crate::tree::{NodeIdx, Tree};
use crate::{aligned_seq, record, Result};
use bio::io::fasta::Record;
use hashbrown::HashMap;

pub struct ParsimonyIndelPoints {}

impl<A: Alignment, AA: AncestralAlignment> AncestralSequenceReconstruction<A, AA>
    for ParsimonyIndelPoints
{
    fn reconstruct_ancestral_seqs(&self, alignment: &A, tree: &Tree) -> Result<AA> {
        let mut aligned_leaf_records = Vec::new();
        for leaf in tree.leaves() {
            let unaligned_record = alignment.seqs().record_by_id(&leaf.id);
            debug_assert_eq!(unaligned_record.id(), leaf.id);
            aligned_leaf_records.push(record!(
                &leaf.id,
                unaligned_record.desc(),
                &aligned_seq!(alignment.leaf_map(&leaf.idx), unaligned_record.seq())
            ));
        }
        aligned_leaf_records.append(&mut get_ancestral_records(tree, alignment));
        let all_seqs = Sequences::new(aligned_leaf_records);
        AA::from_aligned_with_ancestral(all_seqs, tree)
    }
}

/// Infers ancestral sequences for every site independently:
/// - insertion location is latest common ancestors of all non-gap characters
/// - deletion location is node n such that every leaf in the subtree rooted in n
///   is a gap and the parent of n is not a deletion location
fn get_ancestral_records<A: Alignment>(tree: &Tree, alignment: &A) -> Vec<Record> {
    // counter[node] keeps track of the sequence index during the procedurally generated mapping for the node
    let mut ancestral_seqs = HashMap::new();
    // TODO: i could also have a matrix has_char[node, site] and then
    //       loop over nodes and then sites, which might be faster due to caches
    //       see https://github.com/acg-team/rust-phylo/pull/48/files#r2046683347
    for site in 0..alignment.len() {
        // upward pass
        let has_char = get_has_char(alignment, tree, site);
        // downward pass
        elongate_seqs(tree, &mut ancestral_seqs, site, has_char);
    }
    let mut records = Vec::new();
    for (node_idx, seq) in ancestral_seqs {
        records.push(record!(
            tree.node_id(&node_idx),
            None,
            &seq.iter()
                .map(|&b| if b { b'X' } else { GAP })
                .collect::<Vec<u8>>()
        ));
    }
    records
}

fn elongate_seqs(
    tree: &Tree,
    ancestral_seqs: &mut HashMap<NodeIdx, Vec<bool>>,
    site: usize,
    has_char: Vec<bool>,
) {
    for node_idx in tree.preorder() {
        if let Internal(_) = node_idx {
            let children = &tree.node(node_idx).children;
            let parent = &tree.node(node_idx).parent;
            let both_chars =
                has_char[usize::from(children[0])] && has_char[usize::from(children[1])];
            let both_gap =
                !has_char[usize::from(children[0])] && !has_char[usize::from(children[1])];

            let parent_char = match parent {
                None => false,
                Some(parent) => ancestral_seqs[parent][site],
            };
            let current_char = (both_chars || parent_char) && !both_gap;
            // appending to the mapping of the node
            ancestral_seqs
                .entry(*node_idx)
                .or_insert_with(Vec::new)
                .push(current_char);
        }
    }
}

/// Returns a vector V with V[n] = true iff there is a leaf in the subtree rooted in node n that is not a gap.
fn get_has_char<A: Alignment>(leaf_alignment: &A, tree: &Tree, site: usize) -> Vec<bool> {
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

#[cfg(test)]
mod tests;
