use approx::relative_eq;
use log::warn;
use std::fmt::Display;

use crate::evolutionary_models::ProteinModelType;
use crate::substitution_models::{
    blosum_freqs, blosum_q, hivb_freqs, hivb_q, wag_freqs, wag_q, FreqVector, SubstMatrix,
};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProteinParameter {
    Pi,
    Mu,
    Lambda,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProteinSubstParams {
    pub(crate) model_type: ProteinModelType,
    pub(crate) pi: FreqVector,
}

impl ProteinSubstParams {
    pub(crate) fn new(model_type: ProteinModelType, _: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            model_type,
            pi: match model_type {
                ProteinModelType::WAG => wag_freqs(),
                ProteinModelType::HIVB => hivb_freqs(),
                ProteinModelType::BLOSUM => blosum_freqs(),
                _ => unreachable!(),
            },
        })
    }

    pub(crate) fn freqs(&self) -> &FreqVector {
        &self.pi
    }

    pub(crate) fn set_freqs(&mut self, pi: FreqVector) {
        if !relative_eq!(pi.sum(), 1.0, epsilon = 1e-10) {
            warn!("Frequencies must sum to 1.0, not setting values");
        } else {
            self.pi = pi;
        }
    }

    pub(crate) fn q(&self) -> SubstMatrix {
        match self.model_type {
            ProteinModelType::WAG => wag_q(self),
            ProteinModelType::BLOSUM => blosum_q(self),
            ProteinModelType::HIVB => hivb_q(self),
            _ => {
                unreachable!("Protein substitution model should have been defined by now.")
            }
        }
    }
}

impl Display for ProteinSubstParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pi)
    }
}
