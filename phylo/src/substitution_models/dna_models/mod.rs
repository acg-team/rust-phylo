use std::collections::HashMap;

use log::{info, warn};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{DNAModelType, EvoModelInfo, EvoModelParams, EvolutionaryModel};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::sequences::NUCLEOTIDES;
use crate::substitution_models::{
    FreqVector, ParsimonyModel, SubstMatrix, SubstModelInfo, SubstitutionLikelihoodCost,
    SubstitutionModel,
};
use crate::{frequencies, Result, Rounding};

pub(crate) mod dna_substitution_parameters;
pub(crate) use dna_substitution_parameters::*;

pub(crate) mod dna_model_generics;
pub(crate) use dna_model_generics::*;

#[derive(Debug, Clone, PartialEq)]
pub struct DNASubstModel {
    pub(crate) params: DNASubstParams,
    pub(crate) q: SubstMatrix,
}

impl SubstitutionModel for DNASubstModel {
    type ModelType = DNAModelType;
    type Params = DNASubstParams;
    const N: usize = 4;

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

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.params.pi
    }

    fn index(&self) -> &'static [usize; 255] {
        &NUCLEOTIDE_INDEX
    }

    fn get_q(&self) -> &SubstMatrix {
        &self.q
    }

    fn normalise(&mut self) {
        let factor = -(self.params.get_pi().transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }
}

impl EvolutionaryModel for DNASubstModel {
    type ModelType = DNAModelType;
    type Params = DNASubstParams;

    fn new(model: DNAModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        SubstitutionModel::new(model, params)
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        SubstitutionModel::get_p(self, time)
    }

    fn get_q(&self) -> &SubstMatrix {
        SubstitutionModel::get_q(self)
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        SubstitutionModel::get_rate(self, i, j)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        SubstitutionModel::get_stationary_distribution(self)
    }

    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector {
        let mut probs = SubstitutionModel::get_stationary_distribution(self)
            .clone()
            .component_mul(char_encoding);
        probs.scale_mut(1.0 / probs.sum());
        probs
    }

    fn index(&self) -> &[usize; 255] {
        SubstitutionModel::index(self)
    }

    fn get_params(&self) -> &DNASubstParams {
        &self.params
    }
}

impl ParsimonyModel for DNASubstModel {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        SubstitutionModel::generate_scorings(self, times, zero_diag, rounding)
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        SubstitutionModel::get_scoring_matrix(self, time, rounding)
    }
}

pub type DNASubstModelInfo = SubstModelInfo<DNASubstModel>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, DNASubstModel>;

impl<SubstModel: SubstitutionModel> EvoModelInfo for SubstModelInfo<SubstModel> {
    type Model = SubstModel;

    fn new(info: &PhyloInfo, model: &SubstModel) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new(info, model)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl<'a> LikelihoodCostFunction<'a> for DNALikelihoodCost<'a> {
    type Model = DNASubstModel;
    type Info = DNASubstModelInfo;

    fn compute_log_likelihood(&self) -> f64 {
        self.compute_log_likelihood().0
    }

    fn get_empirical_frequencies(&self) -> FreqVector {
        let all_counts = self.info.get_counts();
        let mut total = all_counts.values().sum::<f64>();
        let index = SubstitutionModel::index(self.model);
        let mut freqs = frequencies!(&[0.0; Self::Model::N]);
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
