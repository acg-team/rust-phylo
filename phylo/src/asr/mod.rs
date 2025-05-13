use crate::alignment::{Alignment, AncestralAlignment};
use crate::tree::Tree;
use crate::Result;

pub trait AncestralSequenceReconstruction<A: Alignment, AA: AncestralAlignment> {
    fn reconstruct_ancestral_seqs(&self, leaf_alignment: &A, tree: &Tree) -> Result<AA>;
}
