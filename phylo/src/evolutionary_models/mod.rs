use std::collections::HashMap;

use nalgebra::DVector;
use ordered_float::OrderedFloat;

use crate::substitution_models::FreqVector;
use crate::substitution_models::SubstMatrix;
use crate::Result;

pub trait EvolutionaryModel<const N: usize> {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized;
    fn get_p(&self, time: f64) -> SubstMatrix<N>;
    fn get_rate(&self, i: u8, j: u8) -> f64;
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix<N>, f64)>;
    fn normalise(&mut self);
    fn get_scoring_matrix(&self, time: f64, rounded: bool) -> (SubstMatrix<N>, f64);
    fn get_stationary_distribution(&self) -> &FreqVector<N>;
    fn get_char_probability(&self, char: u8) -> DVector<f64>;
}
