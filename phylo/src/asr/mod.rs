use crate::alignment::{Alignment, AncestralAlignment};
use crate::tree::Tree;
use crate::Result;

pub trait AncestralSequenceReconstruction<A: Alignment, AA: AncestralAlignment> {
    /// Assumes that the nodes of the tree have unique ids.
    fn reconstruct_ancestral_seqs(&self, leaf_alignment: &A, tree: &Tree) -> Result<AA>;
}
