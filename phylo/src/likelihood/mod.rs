use crate::substitution_models::FreqVector;
use crate::tree::{NodeIdx, Tree};

pub trait PhyloCostFunction {
    fn cost(&self) -> f64;
    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]);
    fn tree(&self) -> &Tree;
    fn set_param(&mut self, param: usize, value: f64);
    fn params(&self) -> &[f64];
    fn set_freqs(&mut self, freqs: FreqVector);
    fn empirical_freqs(&self) -> FreqVector;
    fn freqs(&self) -> &FreqVector;
}

#[cfg(test)]
mod tests;
