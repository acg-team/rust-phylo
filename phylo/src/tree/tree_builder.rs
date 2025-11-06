use crate::alignment::Sequences;
use crate::tree::Tree;
use crate::Result;

/// A trait for building phylogenetic trees from a set of sequences.
pub trait TreeBuilder {
    fn build(&mut self, seqs: &Sequences) -> Result<Tree>;
}
