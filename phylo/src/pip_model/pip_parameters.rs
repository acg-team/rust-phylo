use std::fmt::Display;

use anyhow::bail;

use crate::substitution_models::{
    DNAParameter::{self, *},
    DNASubstModel, FreqVector, ProteinParameter, ProteinSubstModel, SubstitutionModel,
};
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PIPParameter {
    Lambda,
    Mu,
    Rtc,
    Rta,
    Rtg,
    Rca,
    Rcg,
    Rag,
}

impl From<PIPParameter> for ProteinParameter {
    fn from(_val: PIPParameter) -> Self {
        unreachable!()
    }
}

impl From<ProteinParameter> for PIPParameter {
    fn from(_val: ProteinParameter) -> Self {
        unreachable!()
    }
}

impl From<PIPParameter> for DNAParameter {
    fn from(val: PIPParameter) -> Self {
        match val {
            PIPParameter::Rtc => Rtc,
            PIPParameter::Rta => Rta,
            PIPParameter::Rtg => Rtg,
            PIPParameter::Rca => Rca,
            PIPParameter::Rcg => Rcg,
            PIPParameter::Rag => Rag,
            _ => unreachable!(),
        }
    }
}

impl From<DNAParameter> for PIPParameter {
    fn from(param: DNAParameter) -> Self {
        match param {
            Rtc => PIPParameter::Rtc,
            Rta => PIPParameter::Rta,
            Rtg => PIPParameter::Rtg,
            Rca => PIPParameter::Rca,
            Rcg => PIPParameter::Rcg,
            Rag => PIPParameter::Rag,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PIPParams<SubstModel: SubstitutionModel> {
    pub(crate) model_type: SubstModel::ModelType,
    pub(crate) subst_model: SubstModel,
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
    PIPParameter: Into<SM::Parameter>,
    SM::Parameter: Into<PIPParameter>,
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

    pub(crate) fn param(&self, param_name: &PIPParameter) -> f64 {
        match param_name {
            PIPParameter::Lambda => self.lambda,
            PIPParameter::Mu => self.mu,
            _ => self.subst_model.param(&(*param_name).into()),
        }
    }

    pub(crate) fn set_param(&mut self, param_name: &PIPParameter, value: f64) {
        match param_name {
            PIPParameter::Lambda => self.lambda = value,
            PIPParameter::Mu => self.mu = value,
            _ => {
                self.subst_model.set_param(&(*param_name).into(), value);
            }
        }
    }
}

impl<SubstModel: SubstitutionModel> Display for PIPParams<SubstModel>
where
    SubstModel: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lambda = {:.5},\nmu = {:.5},\n Substitution model: {}",
            self.lambda, self.mu, self.subst_model
        )
    }
}
