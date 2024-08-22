use log::{info, warn};

use crate::alphabets::NUCLEOTIDE_INDEX;
use crate::evolutionary_models::DNAModelType;
use crate::substitution_models::{
    DNAParameter::*, FreqVector, SubstLikelihoodCost, SubstMatrix, SubstModel, SubstModelInfo,
    SubstitutionModel,
};
use crate::Result;

pub(crate) mod dna_subst_params;
pub(crate) use dna_subst_params::*;
pub(crate) mod dna_generics;
pub(crate) use dna_generics::*;

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

    fn normalise(&mut self) {
        let factor = -(self.params.freqs().transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn model_type(&self) -> &Self::ModelType {
        &self.params.model_type
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn parameter_definition(&self) -> Vec<(&'static str, Vec<DNAParameter>)> {
        match self.params.model_type {
            DNAModelType::JC69 => vec![],
            DNAModelType::K80 => vec![
                ("alpha", vec![Rtc, Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::HKY => vec![
                ("alpha", vec![Rtc, Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::TN93 => vec![
                ("alpha1", vec![Rtc]),
                ("alpha2", vec![Rag]),
                ("beta", vec![Rta, Rtg, Rca, Rcg]),
            ],
            DNAModelType::GTR => vec![
                ("rca", vec![Rca]),
                ("rcg", vec![Rcg]),
                ("rta", vec![Rta]),
                ("rtc", vec![Rtc]),
                ("rtg", vec![Rtg]),
            ],
            _ => unreachable!(),
        }
    }

    fn param(&self, param_name: &DNAParameter) -> f64 {
        self.params.param(param_name)
    }

    fn set_param(&mut self, param_name: &DNAParameter, value: f64) {
        self.params.set_param(param_name, value);
        self.update();
    }

    fn freqs(&self) -> &FreqVector {
        &self.params.pi
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.params.set_freqs(pi);
        self.update();
    }

    fn index(&self) -> &'static [usize; 255] {
        &NUCLEOTIDE_INDEX
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
