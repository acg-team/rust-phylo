use std::fmt::Display;

use crate::substitution_models::dna_models::{DNASubstParams, Parameter};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct PIPDNAParams {
    pub subst_params: DNASubstParams,
    pub lambda: f64,
    pub mu: f64,
}

impl PIPDNAParams {
    pub fn new(subst_params: DNASubstParams, mu: f64, lambda: f64) -> Result<Self> {
        Ok(Self {
            lambda,
            mu,
            subst_params,
        })
    }

    pub fn get_value(&self, param_name: &Parameter) -> f64 {
        match param_name {
            Parameter::Lambda => self.lambda,
            Parameter::Mu => self.mu,
            _ => self.subst_params.get_value(param_name),
        }
    }

    pub fn set_value(&mut self, param_name: &Parameter, value: f64) {
        match param_name {
            Parameter::Lambda => self.lambda = value,
            Parameter::Mu => self.mu = value,
            _ => self.subst_params.set_value(param_name, value),
        }
    }
}

impl Display for PIPDNAParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[lambda = {:.5},\nmu = {:.5},\nsubst model parameters = \n{}]",
            self.lambda, self.mu, self.subst_params
        )
    }
}

impl From<PIPDNAParams> for Vec<f64> {
    fn from(val: PIPDNAParams) -> Self {
        let mut params = vec![val.lambda, val.mu];
        params.append(&mut Vec::<f64>::from(val.subst_params));
        params
    }
}
