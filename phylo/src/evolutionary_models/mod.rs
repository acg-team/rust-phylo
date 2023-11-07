use std::collections::HashMap;

use ordered_float::OrderedFloat;

use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::Result;

impl<const N: usize> std::fmt::Debug for dyn EvolutionaryModel<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EvolutionaryModel with {} states", N)
    }
}

pub trait EvolutionaryModel<const N: usize> {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn get_p(&self, time: f64) -> SubstMatrix;
    fn get_rate(&self, i: u8, j: u8) -> f64;
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)>;
    fn normalise(&mut self);
    fn get_scoring_matrix(&self, time: f64, rounded: bool) -> (SubstMatrix, f64);
    fn get_stationary_distribution(&self) -> &FreqVector;
    fn get_char_probability(&self, char: u8) -> FreqVector;
}

pub trait EvolutionaryModelInfo<const N: usize> {
    fn new(info: &PhyloInfo, model: &dyn EvolutionaryModel<N>) -> Result<Self>
    where
        Self: std::marker::Sized;
}
