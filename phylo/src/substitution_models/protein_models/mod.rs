use log::warn;
use std::fmt::Display;
use std::vec;

use crate::alphabets::AMINOACID_INDEX;
use crate::evolutionary_models::{EvoModelParams, ProteinModelType};
use crate::substitution_models::{
    FreqVector, SubstLikelihoodCost, SubstMatrix, SubstModel, SubstModelInfo, SubstitutionModel,
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
}

impl EvoModelParams for ProteinSubstParams {
    type Parameter = ProteinParameter;

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<ProteinParameter>)> {
        vec![]
    }

    fn param(&self, _param_name: &ProteinParameter) -> f64 {
        0.0
    }
    fn set_param(&mut self, _param_name: &ProteinParameter, _value: f64) {}

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
    type Parameter = ProteinParameter;
    const N: usize = 20;

    fn new(model_type: ProteinModelType, _: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = ProteinSubstParams::new(model_type, &[])?;
        let mut model = ProteinSubstModel::create(&params);
        model.normalise();
        Ok(model)
    }

    fn update(&mut self) {
        let mut model = ProteinSubstModel::create(&self.params);
        model.normalise();
        self.q = model.q;
    }

    fn normalise(&mut self) {
        let factor = -(self.params.freqs().transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<Self::Parameter>)> {
        vec![]
    }

    fn set_param(&mut self, param_name: &Self::Parameter, value: f64) {
        self.params.set_param(param_name, value);
        self.update();
    }

    fn freqs(&self) -> &FreqVector {
        &self.params.pi
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.params.set_freqs(freqs);
        self.update();
    }

    fn param(&self, param_name: &Self::Parameter) -> f64 {
        self.params.param(param_name)
    }

    fn index(&self) -> &'static [usize; 255] {
        &AMINOACID_INDEX
    }
}

impl ProteinSubstModel {
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
}
