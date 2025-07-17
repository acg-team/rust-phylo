use anyhow::bail;

use crate::alignment::{sequence_ids_are_unique, validate_taxa_ids, Alignment, AncestralAlignment};
use crate::tree::{node_ids_are_unique, Tree};
use crate::Result;

pub trait AncestralSequenceReconstruction<A: Alignment, AA: AncestralAlignment> {
    fn reconstruct_ancestral_seqs(&self, leaf_alignment: &A, tree: &Tree) -> Result<AA> {
        if leaf_alignment.seq_count() != tree.n {
            bail!(
                "Alignment has {} sequences, but tree has {} leaves.",
                leaf_alignment.seq_count(),
                tree.n
            );
        }
        node_ids_are_unique(&tree)?;
        sequence_ids_are_unique(leaf_alignment.seqs())?;
        validate_taxa_ids(tree, leaf_alignment.seqs())?;

        Ok(self.reconstruct_ancestral_seqs_unchecked(leaf_alignment, tree))
    }
    fn reconstruct_ancestral_seqs_unchecked(&self, leaf_alignment: &A, tree: &Tree) -> AA;
}
