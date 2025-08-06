use crate::alignment::Sequences;
use crate::tree::Tree;
use crate::Result;

pub trait TreeBuilder {
    fn build(&self, seqs: &Sequences) -> Result<Tree>;
}
