use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::Result;

#[derive(Clone, Copy, Debug)]
pub enum FrequencyOptimisation {
    Empirical,
    Estimated,
    Fixed,
}

pub trait EvoModel {
    fn new(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized;
    fn p(&self, time: f64) -> SubstMatrix;
    fn q(&self) -> &SubstMatrix;
    fn rate(&self, i: u8, j: u8) -> f64;
    fn params(&self) -> &[f64];
    fn set_param(&mut self, param: usize, value: f64);
    fn freqs(&self) -> &FreqVector;
    fn set_freqs(&mut self, pi: FreqVector);
    fn index(&self) -> &[usize; 255];
    fn n(&self) -> usize;
}
