use std::fmt::Display;

use log::warn;

use crate::evolutionary_models::EvolutionaryModelParameters;
use crate::substitution_models::{
    dna_models::DNAModelType::{self, *},
    FreqVector,
};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DNAParameter {
    Pit,
    Pic,
    Pia,
    Pig,
    Rtc,
    Rta,
    Rtg,
    Rca,
    Rcg,
    Rag,
    Mu,
    Lambda,
}
use DNAParameter::*;

use super::{gtr_params, hky_params, jc69_params, k80_params, tn93_params};

#[derive(Clone, Debug, PartialEq)]
pub struct DNASubstParams {
    pub(crate) model_type: DNAModelType,
    pub pi: FreqVector,
    pub rtc: f64,
    pub rta: f64,
    pub rtg: f64,
    pub rca: f64,
    pub rcg: f64,
    pub rag: f64,
}

impl EvolutionaryModelParameters for DNASubstParams {
    type Model = DNAModelType;
    type Parameter = DNAParameter;

    fn new(model_type: &DNAModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        match model_type {
            JC69 => jc69_params(params),
            K80 => k80_params(params),
            HKY => hky_params(params),
            TN93 => tn93_params(params),
            GTR => gtr_params(params),
            _ => unreachable!(),
        }
    }

    fn get_value(&self, param_name: &DNAParameter) -> f64 {
        match param_name {
            Pit => self.pi[0],
            Pic => self.pi[1],
            Pia => self.pi[2],
            Pig => self.pi[3],
            Rtc => self.rtc,
            Rta => self.rta,
            Rtg => self.rtg,
            Rca => self.rca,
            Rcg => self.rcg,
            Rag => self.rag,
            _ => panic!("Invalid parameter name."),
        }
    }

    fn set_value(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Pit | Pic | Pia | Pig => {
                warn!("Cannot set frequencies individually. Use set_pi() instead.")
            }
            Rtc => self.rtc = value,
            Rta => self.rta = value,
            Rtg => self.rtg = value,
            Rca => self.rca = value,
            Rcg => self.rcg = value,
            Rag => self.rag = value,
            _ => panic!("Invalid parameter name."),
        }
    }

    fn set_pi(&mut self, pi: FreqVector) {
        if pi.sum() != 1.0 {
            warn!("Frequencies must sum to 1.0, not setting values");
        } else {
            self.pi = pi;
        }
    }

    fn parameter_definition(model_type: &DNAModelType) -> Vec<(&'static str, Vec<DNAParameter>)> {
        match model_type {
            DNAModelType::JC69 => vec![],
            DNAModelType::K80 => vec![
                ("alpha", vec![Rtc, Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::HKY => vec![
                ("alpha", vec![Rtc, Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::TN93 => vec![
                ("alpha1", vec![Rtc]),
                ("alpha2", vec![Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::GTR => vec![
                ("rca", vec![Rca]),
                ("rcg", vec![Rcg]),
                ("rta", vec![Rta]),
                ("rtc", vec![Rtc]),
                ("rtg", vec![Rtg]),
            ],
            _ => unreachable!(),
        }
    }
}

impl Display for DNASubstParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.print_as(self.model_type))
    }
}

impl DNASubstParams {
    pub(crate) fn print_as(&self, model_type: DNAModelType) -> String {
        match model_type {
            DNAModelType::JC69 => format!("[lambda = {}]", self.rtc),
            DNAModelType::K80 => format!("[alpha = {}, beta = {}]", self.rtc, self.rta),
            DNAModelType::HKY => format!(
                "[pi = {:?}, alpha = {}, beta = {}]",
                self.pi.as_slice(),
                self.rtc,
                self.rta
            ),
            DNAModelType::TN93 => format!(
                "[pi = {:?}, alpha1 = {}, alpha2 = {}, beta = {}]",
                self.pi.as_slice(),
                self.rtc,
                self.rag,
                self.rta
            ),
            DNAModelType::GTR => format!(
                "[pi = {:?}, rtc = {}, rta = {}, rtg = {}, rca = {}, rcg = {}, rag = {}]",
                self.pi.as_slice(),
                self.rtc,
                self.rta,
                self.rtg,
                self.rca,
                self.rcg,
                self.rag
            ),
            _ => unreachable!(),
        }
    }
}

impl From<DNASubstParams> for Vec<f64> {
    fn from(val: DNASubstParams) -> Self {
        vec![
            val.pi[0], val.pi[1], val.pi[2], val.pi[3], val.rtc, val.rta, val.rtg, val.rca,
            val.rcg, val.rag,
        ]
    }
}
