use std::fmt::Display;

use crate::evolutionary_models::{EvoModelParams, ProteinModelType};
use crate::substitution_models::{
    FreqVector, SubstMatrix, SubstModel, SubstModelInfo, SubstitutionLikelihoodCost,
    SubstitutionModel,
};
use crate::Result;

pub(crate) mod protein_model_generics;
pub(crate) use protein_model_generics::*;

pub(crate) type ProteinSubstArray = [f64; 400];
pub(crate) type ProteinFrequencyArray = [f64; 20];

pub type ProteinSubstModel = SubstModel<ProteinSubstParams>;
pub type ProteinSubstModelInfo = SubstModelInfo<ProteinSubstModel>;
pub type ProteinLikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, ProteinSubstModel>;

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

impl EvoModelParams for ProteinSubstParams {
    type ModelType = ProteinModelType;
    type Parameter = ProteinParameter;
    fn new(model_type: &ProteinModelType, _: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            model_type: *model_type,
            pi: match model_type {
                ProteinModelType::WAG => wag_pi(),
                ProteinModelType::HIVB => hivb_pi(),
                ProteinModelType::BLOSUM => blosum_pi(),
                _ => unreachable!(),
            },
        })
    }
    fn parameter_definition(
        _model_type: &ProteinModelType,
    ) -> Vec<(&'static str, Vec<ProteinParameter>)> {
        todo!()
    }
    fn get_value(&self, _param_name: &ProteinParameter) -> f64 {
        todo!()
    }
    fn set_value(&mut self, _param_name: &ProteinParameter, _value: f64) {
        todo!()
    }
    fn get_pi(&self) -> &FreqVector {
        &self.pi
    }
    fn set_pi(&mut self, _pi: FreqVector) {
        todo!()
    }
}

impl Display for ProteinSubstParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.pi)
    }
}

impl From<ProteinSubstParams> for Vec<f64> {
    fn from(val: ProteinSubstParams) -> Self {
        val.pi.as_slice().to_vec()
    }
}

impl SubstitutionModel for ProteinSubstModel {
    type ModelType = ProteinModelType;
    type Params = ProteinSubstParams;
    const N: usize = 20;
    const ALPHABET: &'static [u8] = b"ACDEFGHIKLMNPQRSTVWY";

    fn char_sets() -> &'static [FreqVector] {
        &PROTEIN_SETS
    }

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
        let params = ProteinSubstParams::new(&model_type, &[])?;
        let mut model = ProteinSubstModel::create(&params);
        model.normalise();
        Ok(model)
    }

    fn index(&self) -> &'static [usize; 255] {
        &AMINOACID_INDEX
    }

    fn get_q(&self) -> &SubstMatrix {
        &self.q
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.params.pi
    }

    fn normalise(&mut self) {
        let factor = -(self.params.get_pi().transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        (self.get_q().clone() * time).exp()
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.get_q()[(self.index()[i as usize], self.index()[j as usize])]
    }

    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &crate::Rounding,
    ) -> std::collections::HashMap<ordered_float::OrderedFloat<f64>, (SubstMatrix, f64)> {
        std::collections::HashMap::<crate::f64_h, (SubstMatrix, f64)>::from_iter(times.iter().map(
            |&time| {
                (
                    crate::f64_h::from(time),
                    self.get_scoring_matrix_corrected(time, zero_diag, rounding),
                )
            },
        ))
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &crate::Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix_corrected(time, false, rounding)
    }

    fn get_scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &crate::Rounding,
    ) -> (SubstMatrix, f64) {
        let p = self.get_p(time);
        let mut scores = p.map(|x| -x.ln());
        if rounding.round {
            scores = scores.map(|x| {
                (x * 10.0_f64.powf(rounding.digits as f64)).round()
                    / 10.0_f64.powf(rounding.digits as f64)
            });
        }
        if zero_diag {
            scores.fill_diagonal(0.0);
        }
        let mean = scores.mean();
        (scores, mean)
    }
}
