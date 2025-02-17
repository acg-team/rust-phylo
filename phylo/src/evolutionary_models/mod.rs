use std::fmt::Display;

use dyn_clone::DynClone;

use crate::alphabets::Alphabet;
use crate::substitution_models::{FreqVector, SubstMatrix};

#[derive(Clone, clap::ValueEnum, Debug, Copy)]
pub enum FrequencyOptimisation {
    Empirical,
    Estimated,
    Fixed,
}

pub trait EvoModel: Display + DynClone {
    fn p(&self, time: f64) -> SubstMatrix;
    fn q(&self) -> &SubstMatrix;
    fn rate(&self, i: u8, j: u8) -> f64;
    fn params(&self) -> &[f64];
    fn set_param(&mut self, param: usize, value: f64);
    fn freqs(&self) -> &FreqVector;
    fn set_freqs(&mut self, pi: FreqVector);
    fn n(&self) -> usize;
    fn alphabet(&self) -> &Alphabet;
}

dyn_clone::clone_trait_object!(EvoModel);
