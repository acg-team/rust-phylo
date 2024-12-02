use std::fmt::Display;

use anyhow::bail;

use crate::substitution_models::{DNASubstModel, FreqVector, ProteinSubstModel, SubstitutionModel};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct PIPParams<SM: SubstitutionModel> {
    pub(crate) model_type: SM::ModelType,
    pub(crate) subst_model: SM,
    pub lambda: f64,
    pub mu: f64,
    pub(crate) pi: FreqVector,
}

pub type PIPDNAParams = PIPParams<DNASubstModel>;
pub type PIPProteinParams = PIPParams<ProteinSubstModel>;

fn check_pip_params(params: &[f64]) -> Result<(f64, f64)> {
    if params.len() < 2 {
        bail!("Too few values provided for PIP, 2 values required, lambda and mu.");
    }
    let lambda = params[0];
    let mu = params[1];
    Ok((lambda, mu))
}

impl<SM: SubstitutionModel + Clone> PIPParams<SM>
where
    SM::ModelType: Clone,
{
    pub(crate) fn new(model_type: SM::ModelType, params: &[f64]) -> Result<Self> {
        let (lambda, mu) = check_pip_params(params)?;
        let subst_model = SM::new(model_type.clone(), &params[2..])?;

        let pi = subst_model.freqs().clone().insert_row(SM::N, 0.0);
        Ok(Self {
            model_type,
            lambda,
            mu,
            subst_model,
            pi,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn freqs(&self) -> &FreqVector {
        &self.pi
    }

    pub(crate) fn set_freqs(&mut self, pi: FreqVector) {
        self.pi = pi.clone().insert_row(SM::N, 0.0);
        self.subst_model.set_freqs(pi.clone());
        self.subst_model.update();
    }

    pub(crate) fn set_param(&mut self, param: usize, value: f64) {
        match param {
            0 => self.lambda = value,
            1 => self.mu = value,
            _ => self.subst_model.set_param(param - 2, value),
        }
    }
}

impl<SM: SubstitutionModel> Display for PIPParams<SM>
where
    SM: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lambda = {:.5},\nmu = {:.5},\n Substitution model: {}",
            self.lambda, self.mu, self.subst_model
        )
    }
}
