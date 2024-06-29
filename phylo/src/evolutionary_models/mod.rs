use std::fmt::Display;

use crate::phylo_info::PhyloInfo;
use crate::substitution_models::dna_models::Parameter;
use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::Result;

pub enum FrequencyOptimisation {
    Empirical,
    Estimated,
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum ModelType {
    DNA(DNAModelType),
    Protein(ProteinModelType),
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum DNAModelType {
    JC69,
    K80,
    HKY,
    TN93,
    GTR,
}
impl Display for DNAModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DNAModelType::JC69 => write!(f, "JC69"),
            DNAModelType::K80 => write!(f, "K80"),
            DNAModelType::HKY => write!(f, "HKY"),
            DNAModelType::TN93 => write!(f, "TN93"),
            DNAModelType::GTR => write!(f, "GTR"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum ProteinModelType {
    WAG,
    BLOSUM,
    HIVB,
}

impl Display for ProteinModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProteinModelType::WAG => write!(f, "WAG"),
            ProteinModelType::BLOSUM => write!(f, "BLOSUM"),
            ProteinModelType::HIVB => write!(f, "HIVB"),
        }
    }
}

pub trait EvolutionaryModelParameters<T> {
    fn new(model_type: &T, params: &[f64]) -> Result<Self>
    where
        Self: Sized;
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
    fn new(model_type: ModelType, params: &[f64]) -> Result<Self>
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
