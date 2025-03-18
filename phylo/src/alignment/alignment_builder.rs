use std::collections::HashMap;

use crate::alignment::{Alignment, Sequences};

use crate::parsimony::costs::{GapCost, SimpleCosts};
use crate::parsimony::pars_align_on_tree;
use crate::tree::Tree;
use crate::Result;

pub struct AlignmentBuilder<'a> {
    tree: &'a Tree,
    seqs: Sequences,
}

impl<'a> AlignmentBuilder<'a> {
    pub fn new(tree: &'a Tree, seqs: Sequences) -> AlignmentBuilder<'a> {
        AlignmentBuilder { tree, seqs }
    }

    pub fn build(self) -> Result<Alignment> {
        self.align_unaligned_seqs()
    }

    fn align_unaligned_seqs(self) -> Result<Alignment> {
        let gap = GapCost::new(2.5, 0.5);
        let costs = SimpleCosts::new(1.0, gap);
        let (aligns, _scores) = pars_align_on_tree(&costs, self.tree, self.seqs.clone());
        let mut alignment = Alignment {
            seqs: Sequences::new(Vec::new()),
            leaf_map: HashMap::new(),
            node_map: aligns,
            leaf_encoding: HashMap::new(),
        };
        let leaf_map = alignment.compile_leaf_map(&self.tree.root, self.tree)?;
        alignment.leaf_map = leaf_map;
        alignment.seqs = self.seqs.into_gapless();
        alignment.leaf_encoding = alignment.seqs.generate_leaf_encoding();
        Ok(alignment)
    }
}
