use std::fmt::Display;

use log::warn;

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

impl DNAModelType {
    pub fn get_model_type(model_name: &str) -> Self {
        match model_name.to_uppercase().as_str() {
            "JC69" => DNAModelType::JC69,
            "K80" => DNAModelType::K80,
            "HKY" => DNAModelType::HKY,
            "TN93" => DNAModelType::TN93,
            "GTR" => DNAModelType::GTR,
            _ => {
                warn!("Unknown DNA model requested, defaulting to GTR.");
                DNAModelType::GTR
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

impl ProteinModelType {
    pub fn get_model_type(model_name: &str) -> Self {
        match model_name.to_uppercase().as_str() {
            "WAG" => ProteinModelType::WAG,
            "BLOSUM" => ProteinModelType::BLOSUM,
            "HIVB" => ProteinModelType::HIVB,
            _ => {
                warn!("Unknown DNA model requested, defaulting to WAG.");
                ProteinModelType::WAG
            }
        }
    }
}

pub trait EvolutionaryModelParameters {
    type Model;
    fn new(model: &Self::Model, params: &[f64]) -> Result<Self>
    where
        Self: Sized;
    fn get_value(&self, param_name: &Parameter) -> f64;
    fn set_value(&mut self, param_name: &Parameter, value: f64);
    fn set_pi(&mut self, pi: FreqVector);
}

// TODO: change pi to a row vector
pub trait EvolutionaryModel<const N: usize> {
    type Model;
    fn new(model: Self::Model, params: &[f64]) -> Result<Self>
    where
        Self: Sized;
    fn get_p(&self, time: f64) -> SubstMatrix;
    fn get_rate(&self, i: u8, j: u8) -> f64;
    fn get_stationary_distribution(&self) -> &FreqVector;
    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector;
    fn index() -> &'static [usize; 255];
}

pub trait EvolutionaryModelInfo<const N: usize> {
    type Model;
    fn new(info: &PhyloInfo, model: &Self::Model) -> Result<Self>
    where
        Self: Sized;
    fn reset(&mut self);
}
