use std::collections::HashMap;

use log::{info, warn};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{DNAModelType, EvolutionaryModel, EvolutionaryModelParameters};
use crate::likelihood::LikelihoodCostFunction;
use crate::sequences::NUCLEOTIDES;
use crate::substitution_models::{
    FreqVector, ParsimonyModel, SubstMatrix, SubstitutionLikelihoodCost, SubstitutionModel,
    SubstitutionModelInfo,
};
use crate::{frequencies, Result, Rounding};

mod dna_substitution_parameters;
pub use dna_substitution_parameters::*;

pub(crate) mod dna_model_generics;
pub(crate) use dna_model_generics::*;

pub type DNASubstModel = SubstitutionModel<4>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, 4>;
pub type DNASubstModelInfo = SubstitutionModelInfo<4>;

pub(crate) fn make_dna_model(params: DNASubstParams) -> DNASubstModel {
    info!(
        "Setting up {} with parameters: {}",
        params.model_type, params
    );
    let pi = params.pi.clone();
    let q = match params.model_type {
        DNAModelType::JC69 => jc69_q(),
        DNAModelType::K80 => k80_q(&params),
        DNAModelType::HKY => tn93_q(&params),
        DNAModelType::TN93 => tn93_q(&params),
        DNAModelType::GTR => gtr_q(&params),
        DNAModelType::UNDEF => {
            unreachable!("DNA substitution model should have been defined by now.")
        }
    };
    DNASubstModel {
        index: *NUCLEOTIDE_INDEX,
        q,
        pi,
    }
}

impl DNASubstModel {}

impl EvolutionaryModel<4> for DNASubstModel {
    type Model = DNAModelType;

    fn new(model_type: Self::Model, params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
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
        Ok(make_dna_model(params))
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        self.get_p(time)
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.get_rate(i, j)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        self.get_stationary_distribution()
    }

    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector {
        let mut probs = self
            .get_stationary_distribution()
            .clone()
            .component_mul(char_encoding);
        probs.scale_mut(1.0 / probs.sum());
        probs
    }

    fn index() -> &'static [usize; 255] {
        &NUCLEOTIDE_INDEX
    }
}

impl ParsimonyModel<4> for DNASubstModel {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        self.generate_scorings(times, zero_diag, rounding)
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix(time, rounding)
    }
}

impl<'a> LikelihoodCostFunction<'a, 4> for DNALikelihoodCost<'a> {
    type Model = DNASubstModel;
    type Info = SubstitutionModelInfo<4>;

    fn compute_log_likelihood(&self, model: &Self::Model) -> f64 {
        self.compute_log_likelihood(model).0
    }

    fn get_empirical_frequencies(&self) -> FreqVector {
        let all_counts = self.info.get_counts();
        let mut total = all_counts.values().sum::<f64>();
        let index = &NUCLEOTIDE_INDEX;
        let mut freqs = frequencies!(&[0.0; 4]);
        for (&char, &count) in all_counts.iter() {
            freqs += &DNA_SETS[char as usize].scale(count);
        }
        for &char in NUCLEOTIDES {
            if freqs[index[char as usize]] == 0.0 {
                freqs[index[char as usize]] += 1.0;
                total += 1.0;
            }
        }
        freqs.map(|x| x / total)
    }
}
