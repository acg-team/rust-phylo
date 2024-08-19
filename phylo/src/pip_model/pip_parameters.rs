use std::fmt::Display;

use anyhow::bail;

use crate::evolutionary_models::EvoModelParams;
use crate::substitution_models::dna_models::DNAParameter::{self, *};
use crate::substitution_models::dna_models::DNASubstModel;
use crate::substitution_models::protein_models::{ProteinParameter, ProteinSubstModel};
use crate::substitution_models::{FreqVector, SubstModelParams, SubstitutionModel};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct PIPParams<SubstModel: SubstitutionModel> {
    pub(crate) model_type: SubstModel::ModelType,
    pub subst_params: SubstModel::Params,
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

impl<SubstModel: SubstitutionModel + Clone> PIPParams<SubstModel>
where
    SubstModel::ModelType: Clone,
    SubstModel::Params: EvoModelParams,
{
    pub fn new(model_type: SubstModel::ModelType, params: &[f64]) -> Result<Self> {
        let (lambda, mu) = check_pip_params(params)?;
        let subst_params = SubstModel::Params::new(model_type.clone(), &params[2..])?;
        let pi = EvoModelParams::freqs(&subst_params)
            .clone()
            .insert_row(SubstModel::N, 0.0);
        Ok(Self {
            model_type,
            lambda,
            mu,
            subst_params,
            pi,
        })
    }

    fn freqs(&self) -> &FreqVector {
        &self.pi
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.pi = pi.clone();
        self.subst_params
            .set_freqs(pi.clone().remove_row(SubstModel::N));
    }
}

impl EvoModelParams for PIPDNAParams {
    type Parameter = DNAParameter;

    fn value(&self, param_name: &DNAParameter) -> f64 {
        match param_name {
            Lambda => self.lambda,
            Mu => self.mu,
            _ => self.subst_params.value(param_name),
        }
    }

    fn set_value(&mut self, param_name: &DNAParameter, value: f64) {
        match param_name {
            Lambda => self.lambda = value,
            Mu => self.mu = value,
            _ => self.subst_params.set_value(param_name, value),
        }
    }

    fn freqs(&self) -> &FreqVector {
        self.freqs()
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.set_freqs(pi)
    }

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<DNAParameter>)> {
        self.subst_params
            .parameter_definition()
            .into_iter()
            .chain([
                ("mu", vec![DNAParameter::Mu]),
                ("lambda", vec![DNAParameter::Lambda]),
            ])
            .collect()
    }
}

impl EvoModelParams for PIPProteinParams {
    type Parameter = ProteinParameter;

    fn value(&self, param_name: &ProteinParameter) -> f64 {
        match param_name {
            ProteinParameter::Lambda => self.lambda,
            ProteinParameter::Mu => self.mu,
            _ => self.subst_params.value(param_name),
        }
    }

    fn set_value(&mut self, param_name: &ProteinParameter, value: f64) {
        match param_name {
            ProteinParameter::Lambda => self.lambda = value,
            ProteinParameter::Mu => self.mu = value,
            _ => self.subst_params.set_value(param_name, value),
        }
    }

    fn freqs(&self) -> &FreqVector {
        self.freqs()
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.set_freqs(pi)
    }

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<ProteinParameter>)> {
        self.subst_params
            .parameter_definition()
            .into_iter()
            .chain([
                ("mu", vec![ProteinParameter::Mu]),
                ("lambda", vec![ProteinParameter::Lambda]),
            ])
            .collect()
    }
}

impl<SubstModel: SubstitutionModel> Display for PIPParams<SubstModel>
where
    SubstModel::Params: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[lambda = {:.5},\nmu = {:.5},\nsubst model parameters = \n{:?}]",
            self.lambda, self.mu, self.subst_params
        )
    }
}
