use anyhow::bail;

use crate::alignment::{Alignment, AncestralAlignment};
use crate::phylo_info::validate_taxa_ids;

use crate::tree::Tree;
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
        tree.node_ids_are_unique()?;
        leaf_alignment.seqs().ids_are_unique()?;
        validate_taxa_ids(tree, leaf_alignment.seqs())?;

        Ok(self.reconstruct_ancestral_seqs_unchecked(leaf_alignment, tree))
    }
    fn reconstruct_ancestral_seqs_unchecked(&self, leaf_alignment: &A, tree: &Tree) -> AA;
}
