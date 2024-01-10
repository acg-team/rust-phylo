use argmin::core::CostFunction;

use crate::evolutionary_models::EvolutionaryModelInfo;
use crate::substitution_models::{
    dna_models::{k80_q, DNASubstModel},
    SubstitutionLikelihoodCost, SubstitutionModelInfo,
};
use crate::Result;

struct K80ModelAlphaOptimiser<'a> {
    likelihood_cost: SubstitutionLikelihoodCost<'a, 4>,
    base_model: DNASubstModel,
}

impl CostFunction for K80ModelAlphaOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut model = self.base_model.clone();
        model.q = k80_q(*param, model.params[1]);
        let mut tmp_info = SubstitutionModelInfo::new(self.likelihood_cost.info, &model)?;
        Ok(self
            .likelihood_cost
            .compute_log_likelihood(&model, &mut tmp_info))
    }
}

pub trait LikelihoodCostFunction<'a, const N: usize> {
    type Model;
    type Info;
    fn compute_log_likelihood(&self, model: &Self::Model) -> f64;
    fn get_empirical_frequencies(&self) -> FreqVector;
}

#[cfg(test)]
mod likelihood_tests;
