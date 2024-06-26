use crate::phylo_info::PhyloInfo;
use crate::substitution_models::dna_models::{DNAModelType, Parameter};
use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::Result;

pub enum FrequencyOptimisation {
    Empirical,
    Estimated,
    Fixed,
}

pub trait EvolutionaryModelParameters {
    fn new(model_type: &DNAModelType, params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn get_value(&self, param_name: &Parameter) -> f64;
    fn set_value(&mut self, param_name: &Parameter, value: f64);
    fn set_pi(&mut self, pi: FreqVector);
}

impl<const N: usize> std::fmt::Debug for dyn EvolutionaryModel<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EvolutionaryModel with {} states", N)
    }
}
// TODO: change pi to a row vector
pub trait EvolutionaryModel<const N: usize> {
    fn new(model_name: &str, params: &[f64]) -> Result<Self>
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
