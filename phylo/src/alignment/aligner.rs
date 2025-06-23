use crate::alignment::{Alignment, Sequences};
use crate::tree::Tree;
use crate::Result;

pub trait Aligner<A: Alignment> {
    fn align(&self, seqs: &Sequences, tree: &Tree) -> Result<A>;
}
