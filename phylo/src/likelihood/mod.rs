use crate::substitution_models::FreqVector;
use crate::tree::{NodeIdx, Tree};

pub trait TreeSearchCost {
    fn cost(&self) -> f64;
    // update_tree implies that the tree is a valid modification of the existing tree (e.g. an SPR move),
    // and that the dirty_nodes are the nodes that have changed, but this is not enforced by the trait.
    // TODO: enforce this in the trait.
    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]);
    fn tree(&self) -> &Tree;
}

pub trait ModelSearchCost {
    fn cost(&self) -> f64;
    fn set_param(&mut self, param: usize, value: f64);
    fn params(&self) -> &[f64];
    fn set_freqs(&mut self, freqs: FreqVector);
    fn empirical_freqs(&self) -> FreqVector;
    fn freqs(&self) -> &FreqVector;
}

#[cfg(test)]
#[cfg_attr(coverage, no_coverage)]
mod tests;
