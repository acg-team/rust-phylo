use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::Result;

impl<const N: usize> std::fmt::Debug for dyn EvolutionaryModel<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EvolutionaryModel with {} states", N)
    }
}

// TODO: change pi to a row vector
pub trait EvolutionaryModel<const N: usize> {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn get_p(&self, time: f64) -> SubstMatrix;
    fn get_rate(&self, i: u8, j: u8) -> f64;
    fn get_stationary_distribution(&self) -> &FreqVector;
    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector;
}

pub trait EvolutionaryModelInfo<const N: usize> {
    fn new(info: &PhyloInfo, model: &dyn EvolutionaryModel<N>) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn reset(&mut self);
}
