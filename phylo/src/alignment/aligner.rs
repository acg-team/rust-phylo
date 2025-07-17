use anyhow::bail;

use crate::alignment::{Alignment, Sequences};
use crate::tree::Tree;
use crate::Result;

use crate::phylo_info::validate_taxa_ids;

pub trait Aligner<A: Alignment> {
    fn align(&self, seqs: &Sequences, tree: &Tree) -> Result<A> {
        if seqs.aligned {
            bail!("Sequences must not be aligned.");
        }
        seqs.ids_are_unique()?;
        validate_taxa_ids(tree, seqs)?;
        Ok(self.align_unchecked(seqs, tree))
    }
    fn align_unchecked(&self, seqs: &Sequences, tree: &Tree) -> A;
}
