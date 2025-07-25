use crate::substitution_models::FreqVector;
use crate::tree::{NodeIdx, Tree};

pub trait TreeSearchCost {
    // The cost function definition for tree search, e.g. the likelihood of the alignment given the model and the tree.
    // The optimisers will maximise the cost, so if the cost should be minimised instead, it should be negated.
    // The likelihood or the log-likelihood are maximised, the parsimony score is minimised.
    fn cost(&self) -> f64;
    // update_tree implies that the tree is a valid modification of the existing tree (e.g. an SPR move),
    // and that the dirty_nodes are the nodes that have changed, but this is not enforced by the trait.
    // TODO: enforce this in the trait. We could define a update_tree method and define default
    // implementation that does some checks and then define a update_tree_unchecked that each
    // implementor must implement.
    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]);
    fn tree(&self) -> &Tree;
    fn blen_optimisation(&self) -> bool {
        true
    }
}

pub trait ModelSearchCost {
    // The cost function definition for model search, e.g. the likelihood of the alignment given the model and the tree.
    // The optimisers will maximise the cost, so if the cost should be minimised instead, it should be negated.
    // The likelihood or the log-likelihood are maximised, the parsimony score is minimised.
    fn cost(&self) -> f64;
    fn set_param(&mut self, param: usize, value: f64);
    fn params(&self) -> &[f64];
    fn set_freqs(&mut self, freqs: FreqVector);
    fn empirical_freqs(&self) -> FreqVector;
    fn freqs(&self) -> &FreqVector;
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
