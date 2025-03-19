use std::collections::HashMap;

use crate::alignment::{Alignment, Sequences};
use crate::parsimony::costs::{GapCost, ParsimonyCosts, SimpleCosts};
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
        let costs = SimpleCosts::new(1.0, GapCost::new(2.5, 0.5));
        let (aligns, _scores) = pars_align_on_tree(&costs, self.tree, self.seqs.clone());

        let mut alignment = Alignment {
            seqs: self.seqs.into_gapless(),
            leaf_map: HashMap::new(),
            node_map: aligns,
            leaf_encoding: HashMap::new(),
        };
        let leaf_map = alignment
            .compile_leaf_map(&self.tree.root, self.tree)
            .unwrap();
        alignment.leaf_map = leaf_map;
        alignment.leaf_encoding = alignment.seqs.generate_leaf_encoding();
        Ok(alignment)
    }

    pub fn build_with_costs(self, costs: &dyn ParsimonyCosts) -> Result<Alignment> {
        let (aligns, _scores) = pars_align_on_tree(costs, self.tree, self.seqs.clone());
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
