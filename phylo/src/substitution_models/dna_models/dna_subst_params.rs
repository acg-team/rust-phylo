use std::fmt::Display;

use log::{info, warn};

use crate::evolutionary_models::DNAModelType::{self, *};
use crate::substitution_models::{
    gtr_params, hky_params, jc69_params, k80_params, tn93_params, FreqVector,
};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DNAParameter {
    Rtc,
    Rta,
    Rtg,
    Rca,
    Rcg,
    Rag,
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

impl DNASubstParams {
    pub(crate) fn param(&self, param_name: &DNAParameter) -> f64 {
        match param_name {
            Rtc => self.rtc,
            Rta => self.rta,
            Rtg => self.rtg,
            Rca => self.rca,
            Rcg => self.rcg,
            Rag => self.rag,
        }
    }

    pub(crate) fn set_param(&mut self, param_name: &DNAParameter, value: f64) {
        match self.model_type {
            JC69 => self.set_jc69_param(param_name, value),
            K80 => self.set_k80_param(param_name, value),
            HKY => self.set_hky_param(param_name, value),
            TN93 => self.set_tn93_param(param_name, value),
            GTR => self.set_gtr_param(param_name, value),
            _ => unreachable!(),
        }
    }

    fn set_jc69_param(&mut self, _: &DNAParameter, value: f64) {
        self.rtc = value;
        self.rta = value;
        self.rtg = value;
        self.rca = value;
        self.rcg = value;
        self.rag = value;
    }

    fn set_k80_param(&mut self, param_name: &DNAParameter, value: f64) {
        self.set_hky_param(param_name, value)
    }

    fn set_hky_param(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Rtc | Rag => {
                self.rtc = value;
                self.rag = value;
            }
            Rta | Rtg | Rca | Rcg => {
                self.rta = value;
                self.rtg = value;
                self.rca = value;
                self.rcg = value;
            }
        }
    }

    fn set_tn93_param(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Rtc => self.rtc = value,
            Rag => self.rag = value,
            Rta | Rtg | Rca | Rcg => {
                self.rta = value;
                self.rtg = value;
                self.rca = value;
                self.rcg = value;
            }
        }
    }

    fn set_gtr_param(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Rtc => self.rtc = value,
            Rta => self.rta = value,
            Rtg => self.rtg = value,
            Rca => self.rca = value,
            Rcg => self.rcg = value,
            Rag => self.rag = value,
        }
    }

    pub(crate) fn freqs(&self) -> &FreqVector {
        &self.pi
    }

    pub(crate) fn set_freqs(&mut self, pi: FreqVector) {
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
}

impl Display for DNASubstParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.model_type {
            DNAModelType::JC69 => write!(f, "[lambda = {}]", self.rtc),
            DNAModelType::K80 => write!(f, "[alpha = {}, beta = {}]", self.rtc, self.rta),
            DNAModelType::HKY => write!(
                f,
                "[pi = {:?}, alpha = {}, beta = {}]",
                self.pi.as_slice(),
                self.rtc,
                self.rta
            ),
            DNAModelType::TN93 => write!(
                f,
                "[pi = {:?}, alpha1 = {}, alpha2 = {}, beta = {}]",
                self.pi.as_slice(),
                self.rtc,
                self.rag,
                self.rta
            ),
            DNAModelType::GTR => write!(
                f,
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
