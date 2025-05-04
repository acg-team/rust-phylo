use crate::alignment::{AlignmentTrait, AncestralAlignmentTrait};
use crate::tree::Tree;
use crate::Result;

pub trait Asr {
    fn asr<L: AlignmentTrait>(
        &self,
        leaf_alignment: &L,
        tree: &Tree,
    ) -> Result<impl AncestralAlignmentTrait>;
}
