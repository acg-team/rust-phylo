use log::warn;
use std::fmt::Display;

use crate::alphabets::AMINOACID_INDEX;
use crate::evolutionary_models::{EvoModelParams, ProteinModelType};
use crate::substitution_models::{
    FreqVector, SubstLikelihoodCost, SubstMatrix, SubstModel, SubstModelInfo, SubstModelParams,
    SubstitutionModel,
};
use crate::Result;

pub(crate) mod protein_model_generics;
pub(crate) use protein_model_generics::*;
pub(crate) type ProteinSubstArray = [f64; 400];
pub(crate) type ProteinFrequencyArray = [f64; 20];

pub type ProteinSubstModel = SubstModel<ProteinSubstParams>;
pub type ProteinSubstModelInfo = SubstModelInfo<ProteinSubstModel>;
pub type ProteinLikelihoodCost<'a> = SubstLikelihoodCost<'a, ProteinSubstModel>;

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

impl SubstModelParams for ProteinSubstParams {
    type ModelType = ProteinModelType;

    fn new(model_type: ProteinModelType, _: &[f64]) -> Result<Self>
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
}

impl EvoModelParams for ProteinSubstParams {
    type Parameter = ProteinParameter;

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<ProteinParameter>)> {
        vec![]
    }
    fn value(&self, _param_name: &ProteinParameter) -> f64 {
        0.0
    }
    fn set_value(&mut self, _param_name: &ProteinParameter, _value: f64) {}
    fn freqs(&self) -> &FreqVector {
        &self.pi
    }
    fn set_freqs(&mut self, pi: FreqVector) {
        if pi.sum() != 1.0 {
            warn!("Frequencies must sum to 1.0, not setting values");
        } else {
            self.pi = pi;
        }
    }
}

impl Display for ProteinSubstParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pi)
    }
}

impl SubstitutionModel for ProteinSubstModel {
    type ModelType = ProteinModelType;
    type Params = ProteinSubstParams;
    const N: usize = 20;

    fn create(params: &ProteinSubstParams) -> ProteinSubstModel {
        let q = match params.model_type {
            ProteinModelType::WAG => wag_q(),
            ProteinModelType::BLOSUM => blosum_q(),
            ProteinModelType::HIVB => hivb_q(),
            _ => {
                unreachable!("Protein substitution model should have been defined by now.")
            }
        };
        ProteinSubstModel {
            q,
            params: params.clone(),
        }
    }

    fn new(model_type: ProteinModelType, _: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = ProteinSubstParams::new(model_type, &[])?;
        let mut model = ProteinSubstModel::create(&params);
        model.normalise();
        Ok(model)
    }

    fn index(&self) -> &'static [usize; 255] {
        &AMINOACID_INDEX
    }

    fn freqs(&self) -> &FreqVector {
        &self.params.pi
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn normalise(&mut self) {
        let factor = -(self.params.freqs().transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }
}
