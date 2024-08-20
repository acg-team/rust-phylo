use log::{info, warn};

use crate::alphabets::NUCLEOTIDE_INDEX;
use crate::evolutionary_models::{DNAModelType, EvoModelParams};
use crate::substitution_models::{
    FreqVector, SubstLikelihoodCost, SubstMatrix, SubstModel, SubstModelInfo, SubstitutionModel,
};
use crate::Result;

pub(crate) mod dna_substitution_parameters;
pub(crate) use dna_substitution_parameters::*;

pub(crate) mod dna_model_generics;
pub(crate) use dna_model_generics::*;

pub type DNASubstModel = SubstModel<DNASubstParams>;
pub type DNASubstModelInfo = SubstModelInfo<DNASubstModel>;
pub type DNALikelihoodCost<'a> = SubstLikelihoodCost<'a, DNASubstModel>;

impl SubstitutionModel for DNASubstModel {
    type ModelType = DNAModelType;
    type Parameter = DNAParameter;

    const N: usize = 4;

    fn new(model_type: DNAModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = if model_type == DNAModelType::UNDEF {
            warn!("No model provided, defaulting to GTR.");
            DNASubstParams::new(
                DNAModelType::GTR,
                [0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(),
            )?
        } else {
            DNASubstParams::new(model_type, params)?
        };
        info!(
            "Setting up {} with parameters: {}",
            params.model_type, params
        );
        Ok(DNASubstModel::create(&params))
    }

    fn freqs(&self) -> &FreqVector {
        &self.params.pi
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.params.set_freqs(pi);
        self.update()
    }

    fn index(&self) -> &'static [usize; 255] {
        &NUCLEOTIDE_INDEX
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn normalise(&mut self) {
        let factor = -(self.params.freqs().transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn update(&mut self) {
        let q = match self.params.model_type {
            DNAModelType::JC69 => jc69_q(),
            DNAModelType::K80 => k80_q(&self.params),
            DNAModelType::HKY => tn93_q(&self.params),
            DNAModelType::TN93 => tn93_q(&self.params),
            DNAModelType::GTR => gtr_q(&self.params),
            DNAModelType::UNDEF => {
                unreachable!("DNA substitution model should have been defined by now.")
            }
        };
        self.q = q;
    }

    fn param(&self, param_name: &Self::Parameter) -> f64 {
        self.params.param(param_name)
    }

    fn set_param(&mut self, param_name: &Self::Parameter, value: f64) {
        self.params.set_param(param_name, value);
        self.update();
    }

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<Self::Parameter>)> {
        self.params.parameter_definition()
    }
}

impl DNASubstModel {
    pub(crate) fn create(params: &DNASubstParams) -> DNASubstModel {
        let q = match params.model_type {
            DNAModelType::JC69 => jc69_q(),
            DNAModelType::K80 => k80_q(params),
            DNAModelType::HKY => tn93_q(params),
            DNAModelType::TN93 => tn93_q(params),
            DNAModelType::GTR => gtr_q(params),
            DNAModelType::UNDEF => {
                unreachable!("DNA substitution model should have been defined by now.")
            }
        };
        DNASubstModel {
            params: params.clone(),
            q,
        }
    }
}
