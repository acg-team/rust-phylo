use crate::alignment::Sequences;
use crate::tree::Tree;
use crate::Result;

pub trait TreeBuilder {
    fn build_tree(&self, seqs: &Sequences) -> Result<Tree>;
}
