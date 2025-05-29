use std::fmt::Display;

use dyn_clone::DynClone;
use nalgebra::DMatrixViewMut;

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
    fn p_to(&self, time: f64, to: &mut DMatrixViewMut<f64>);
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
