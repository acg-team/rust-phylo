use std::fmt::Display;

use crate::evolutionary_models::{DNAModelType, EvolutionaryModelParameters};
use crate::substitution_models::dna_models::{
    DNAParameter::{self, *},
    DNASubstParams,
};
use crate::substitution_models::FreqVector;
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct PIPDNAParams {
    pub(crate) model_type: DNAModelType,
    pub subst_params: DNASubstParams,
    pub lambda: f64,
    pub mu: f64,
}

impl EvolutionaryModelParameters for PIPDNAParams {
    type Model = DNAModelType;
    type Parameter = DNAParameter;

    fn new(model_type: &DNAModelType, params: &[f64]) -> Result<Self> {
        let lambda = params[0];
        let mu = params[1];
        let subst_params = DNASubstParams::new(model_type, &params[2..])?;
        Ok(Self {
            model_type: *model_type,
            lambda,
            mu,
            subst_params,
        })
    }

    fn get_value(&self, param_name: &DNAParameter) -> f64 {
        match param_name {
            Lambda => self.lambda,
            Mu => self.mu,
            _ => self.subst_params.get_value(param_name),
        }
    }

    fn set_value(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Lambda => self.lambda = value,
            Mu => self.mu = value,
            _ => self.subst_params.set_value(param_name, value),
        }
    }

    fn set_pi(&mut self, pi: FreqVector) {
        self.subst_params.set_pi(pi);
    }

    fn parameter_definition(model_type: &DNAModelType) -> Vec<(&'static str, Vec<DNAParameter>)> {
        DNASubstParams::parameter_definition(model_type)
            .into_iter()
            .chain([
                ("mu", vec![DNAParameter::Mu]),
                ("lambda", vec![DNAParameter::Lambda]),
            ])
            .collect()
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
