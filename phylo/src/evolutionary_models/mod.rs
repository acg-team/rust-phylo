use std::fmt::Display;

use log::warn;

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
    UNDEF,
}

impl Display for DNAModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DNAModelType::JC69 => write!(f, "JC69"),
            DNAModelType::K80 => write!(f, "K80"),
            DNAModelType::HKY => write!(f, "HKY"),
            DNAModelType::TN93 => write!(f, "TN93"),
            DNAModelType::GTR => write!(f, "GTR"),
            DNAModelType::UNDEF => write!(f, "Undefined"),
        }
    }
}

impl From<&str> for DNAModelType {
    fn from(value: &str) -> Self {
        match value.to_uppercase().as_str() {
            "JC69" => DNAModelType::JC69,
            "K80" => DNAModelType::K80,
            "HKY" => DNAModelType::HKY,
            "TN93" => DNAModelType::TN93,
            "GTR" => DNAModelType::GTR,
            _ => {
                warn!("Unknown DNA model {value:?} requested");
                DNAModelType::UNDEF
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum ProteinModelType {
    WAG,
    BLOSUM,
    HIVB,
    UNDEF,
}

impl Display for ProteinModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProteinModelType::WAG => write!(f, "WAG"),
            ProteinModelType::BLOSUM => write!(f, "BLOSUM"),
            ProteinModelType::HIVB => write!(f, "HIVB"),
            ProteinModelType::UNDEF => write!(f, "UNDEF"),
        }
    }
}

impl From<&str> for ProteinModelType {
    fn from(value: &str) -> Self {
        match value.to_uppercase().as_str() {
            "WAG" => ProteinModelType::WAG,
            "BLOSUM" => ProteinModelType::BLOSUM,
            "HIVB" => ProteinModelType::HIVB,
            _ => {
                warn!("Unknown protein model {value:?} requested");
                ProteinModelType::UNDEF
            }
        }
    }
}

// TODO: change pi to a row vector
pub trait EvoModel {
    type Parameter;
    type ModelType;
    const N: usize;

    fn new(model_type: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized;
    fn model_type(&self) -> &Self::ModelType;
    fn description(&self) -> String;
    fn p(&self, time: f64) -> SubstMatrix;
    fn q(&self) -> &SubstMatrix;
    fn rate(&self, i: u8, j: u8) -> f64;
    fn parameter_definition(&self) -> Vec<(&'static str, Vec<Self::Parameter>)>;
    fn param(&self, param_name: &Self::Parameter) -> f64;
    fn set_param(&mut self, param_name: &Self::Parameter, value: f64);
    fn freqs(&self) -> &FreqVector;
    fn set_freqs(&mut self, pi: FreqVector);
    fn index(&self) -> &[usize; 255];
}

#[cfg(test)]
pub(crate) mod tests;
