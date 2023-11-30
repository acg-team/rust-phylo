use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::substitution_models::dna_models::K80Params;
use crate::substitution_models::{
    dna_models::DNASubstModel, SubstitutionLikelihoodCost, SubstitutionModelInfo,
};
use crate::Result;

struct K80ModelAlphaOptimiser<'a> {
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    base_model: &'a DNASubstModel,
}

struct K80ModelBetaOptimiser<'a> {
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    base_model: &'a DNASubstModel,
}

pub struct K80ModelOptimiser<'a> {
    likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    base_model: &'a DNASubstModel,
}

impl<'a> K80ModelOptimiser<'a> {
    pub fn new(
        likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
        base_model: &'a DNASubstModel,
    ) -> Self {
        K80ModelOptimiser {
            likelihood_cost,
            base_model,
        }
    }

    pub fn optimise_parameters(&self) -> Result<(u32, K80Params, f64)> {
        let epsilon = 1e-10;
        let alpha = self.base_model.params[0];
        let beta = self.base_model.params[1];
        let mut model = DNASubstModel::new("k80", &[alpha, beta])?;
        let mut logl = f64::NEG_INFINITY;
        let mut new_logl = 0.0;
        let mut k80_params = K80Params { alpha, beta };
        let mut iters = 0;
        while (logl - new_logl).abs() > epsilon {
            logl = new_logl;
            _ = self.optimise_alpha(&mut model, &mut k80_params)?;
            new_logl = self.optimise_beta(&mut model, &mut k80_params)?;
            iters += 1;
        }
        Ok((iters, k80_params, -logl))
    }

    fn optimise_alpha(&self, model: &mut DNASubstModel, k80_params: &mut K80Params) -> Result<f64> {
        model.reset_k80_q(k80_params);
        let alpha_optimiser = K80ModelAlphaOptimiser {
            likelihood_cost: self.likelihood_cost,
            base_model: model,
        };
        let res = Executor::new(alpha_optimiser, subst_param_brent()).run()?;
        k80_params.alpha = res.state().best_param.unwrap();
        Ok(res.state().best_cost)
    }

    fn optimise_beta(&self, model: &mut DNASubstModel, k80_params: &mut K80Params) -> Result<f64> {
        model.reset_k80_q(k80_params);
        let beta_optimiser = K80ModelBetaOptimiser {
            likelihood_cost: self.likelihood_cost,
            base_model: model,
        };
        let res = Executor::new(beta_optimiser, subst_param_brent()).run()?;
        k80_params.beta = res.state().best_param.unwrap();
        Ok(res.state().best_cost)
    }
}

fn subst_param_brent() -> BrentOpt<f64> {
    BrentOpt::new(1e-10, 100.0)
}

impl CostFunction for K80ModelAlphaOptimiser<'_> {
    type Param = f64;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut model = self.base_model.clone();
        model.reset_k80_q(&K80Params {
            alpha: *param,
            beta: self.base_model.params[1],
        });
        let mut tmp_info = SubstitutionModelInfo::new(self.likelihood_cost.info, &model)?;
        Ok(-self
            .likelihood_cost
            .compute_log_likelihood(&model, &mut tmp_info))
    }
}

impl CostFunction for K80ModelBetaOptimiser<'_> {
    type Param = f64;
    type Output = f64;
    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut model = self.base_model.clone();
        model.reset_k80_q(&K80Params {
            alpha: self.base_model.params[0],
            beta: *param,
        });
        let mut tmp_info = SubstitutionModelInfo::new(self.likelihood_cost.info, &model)?;
        Ok(-self
            .likelihood_cost
            .compute_log_likelihood(&model, &mut tmp_info))
    }
}

pub trait LikelihoodCostFunction<'a, const N: usize> {
    type Model;
    type Info;

    fn compute_log_likelihood(&self, model: &Self::Model, tmp_info: &mut Self::Info) -> f64;
}

#[cfg(test)]
mod likelihood_tests;
