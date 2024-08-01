use log::{info, warn};

use crate::evolutionary_models::{DNAModelType, EvoModelParams};
use crate::substitution_models::{
    FreqVector, SubstMatrix, SubstModel, SubstModelInfo, SubstitutionLikelihoodCost,
    SubstitutionModel,
};
use crate::Result;

pub(crate) mod dna_substitution_parameters;
pub(crate) use dna_substitution_parameters::*;

pub(crate) mod dna_model_generics;
pub(crate) use dna_model_generics::*;

pub type DNASubstModel = SubstModel<DNASubstParams>;
pub type DNASubstModelInfo = SubstModelInfo<DNASubstModel>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, DNASubstModel>;

impl SubstitutionModel for DNASubstModel {
    type ModelType = DNAModelType;
    type Params = DNASubstParams;
    const N: usize = 4;
    const ALPHABET: &'static [u8] = b"TCAG";

    fn char_sets() -> &'static [FreqVector] {
        &DNA_SETS
    }

    fn create(params: &DNASubstParams) -> DNASubstModel {
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

    fn new(model_type: DNAModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = if model_type == DNAModelType::UNDEF {
            warn!("No model provided, defaulting to GTR.");
            DNASubstParams::new(
                &DNAModelType::GTR,
                [0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0].as_slice(),
            )?
        } else {
            DNASubstParams::new(&model_type, params)?
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
}
