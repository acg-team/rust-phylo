use std::fmt::Display;

use log::{info, warn};

use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvoModelParams,
};
use crate::substitution_models::{
    gtr_params, hky_params, jc69_params, k80_params, tn93_params, FreqVector,
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

#[derive(Clone, Debug, PartialEq)]
pub struct DNASubstParams {
    pub(crate) model_type: DNAModelType,
    pub(crate) pi: FreqVector,
    pub(crate) rtc: f64,
    pub(crate) rta: f64,
    pub(crate) rtg: f64,
    pub(crate) rca: f64,
    pub(crate) rcg: f64,
    pub(crate) rag: f64,
}

impl DNASubstParams {
    pub(crate) fn new(model_type: DNAModelType, params: &[f64]) -> Result<Self>
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
}

impl EvoModelParams for DNASubstParams {
    type Parameter = DNAParameter;

    fn param(&self, param_name: &DNAParameter) -> f64 {
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

    fn set_param(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Pit | Pic | Pia | Pig => {
                warn!("Cannot set frequencies individually. Use set_freqs() instead.")
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

    fn freqs(&self) -> &FreqVector {
        &self.pi
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        match self.model_type {
            JC69 | K80 => {
                info!("Model does not have frequency parameters.")
            }
            HKY | TN93 | GTR => {
                if pi.sum() != 1.0 {
                    warn!("Frequencies must sum to 1.0, not setting values");
                } else {
                    self.pi = pi;
                }
            }
            _ => unreachable!(),
        }
    }

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<DNAParameter>)> {
        match self.model_type {
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
