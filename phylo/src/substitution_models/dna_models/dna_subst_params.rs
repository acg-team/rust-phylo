use std::fmt::Display;

use log::{info, warn};

use crate::evolutionary_models::DNAModelType::{self, *};
use crate::substitution_models::{
    gtr_params, hky_params, jc69_params, k80_params, tn93_params, FreqVector,
};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct DNASubstParams {
    // order always RTC, RTA, RTG, RCA, RCG, RAG
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
    pub(crate) fn model_parameters(&self) -> Vec<f64> {
        match self.model_type {
            DNAModelType::JC69 => vec![],
            DNAModelType::K80 => vec![self.rtc, self.rta],
            DNAModelType::HKY => vec![self.rtc, self.rta],
            DNAModelType::TN93 => vec![self.rtc, self.rag, self.rta],
            DNAModelType::GTR => vec![self.rtc, self.rta, self.rtg, self.rca, self.rcg],
            _ => unreachable!(),
        }
    }

    // pub(crate) fn param(&self, param: usize) -> f64 {
    //     match self.model_type {
    //         DNAModelType::JC69 => self.rtc,
    //         DNAModelType::K80 | DNAModelType::HKY => match param {
    //             0 => self.rtc,
    //             1 => self.rta,
    //             _ => {
    //                 unreachable!();
    //             }
    //         },
    //         DNAModelType::TN93 => match param {
    //             0 => self.rtc,
    //             1 => self.rag,
    //             2 => self.rta,
    //             _ => {
    //                 unreachable!();
    //             }
    //         },
    //         DNAModelType::GTR => match param {
    //             0 => self.rtc,
    //             1 => self.rta,
    //             2 => self.rtg,
    //             3 => self.rca,
    //             4 => self.rcg,
    //             _ => {
    //                 unreachable!();
    //             }
    //         },
    //         _ => unreachable!(),
    //     }
    // }

    pub(crate) fn set_param(&mut self, param: usize, value: f64) {
        match self.model_type {
            JC69 => {}
            K80 => self.set_k80_param(param, value),
            HKY => self.set_hky_param(param, value),
            TN93 => self.set_tn93_param(param, value),
            GTR => self.set_gtr_param(param, value),
            _ => unreachable!(),
        }
    }

    fn set_k80_param(&mut self, param: usize, value: f64) {
        self.set_hky_param(param, value);
    }

    fn set_hky_param(&mut self, param: usize, value: f64) {
        match param {
            0 => {
                self.rtc = value;
                self.rag = value;
            }
            1 => {
                self.rta = value;
                self.rtg = value;
                self.rca = value;
                self.rcg = value;
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn set_tn93_param(&mut self, param: usize, value: f64) {
        match param {
            0 => self.rtc = value,
            1 => self.rag = value,
            2 => {
                self.rta = value;
                self.rtg = value;
                self.rca = value;
                self.rcg = value;
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn set_gtr_param(&mut self, param: usize, value: f64) {
        match param {
            0 => self.rtc = value,
            1 => self.rta = value,
            2 => self.rtg = value,
            3 => self.rca = value,
            4 => self.rcg = value,
            _ => {
                unreachable!();
            }
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
