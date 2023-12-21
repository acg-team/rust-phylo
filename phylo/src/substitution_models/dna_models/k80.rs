use std::ops::Div;

use anyhow::bail;
use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;
use log::{info, warn};

use crate::evolutionary_models::EvolutionaryModelInfo;
use crate::substitution_models::SubstParams;
use crate::substitution_models::{
    dna_models::{dna_substitution_parameters::DNASubstParams, make_dna_model, DNASubstModel},
    FreqVector, SubstMatrix, SubstitutionLikelihoodCost, SubstitutionModelInfo,
};
use crate::Result;

pub fn k80(k80_params: DNASubstParams) -> DNASubstModel {
    info!(
        "Setting up k80 with parameters: {}",
        k80_params.print_as_k80()
    );
    let q = k80_q(&k80_params);
    make_dna_model(k80_params, q)
}

pub fn parse_k80_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
    let (alpha, beta) = if model_params.is_empty() {
        warn!("Too few values provided for K80, required 1 or 2 values, kappa or alpha and beta.");
        warn!("Falling back to default values.");
        (2.0, 1.0)
    } else if model_params.len() == 1 {
        (model_params[0], 1.0)
    } else if model_params.len() == 2 {
        (model_params[0], model_params[1])
    } else {
        warn!("Too many values provided for K80, required 2 values, alpha and beta.");
        warn!("Will only use the first two values provided.");
        (model_params[0], model_params[1])
    };
    Ok(k80_params(alpha, beta))
}

fn k80_params(alpha: f64, beta: f64) -> DNASubstParams {
    DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.25; 4]),
        rtc: alpha,
        rta: beta,
        rtg: beta,
        rca: beta,
        rcg: beta,
        rag: alpha,
    }
}

pub fn k80_q(p: &DNASubstParams) -> SubstMatrix {
    let alpha = p.rtc;
    let beta = p.rta;
    let total = alpha + 2.0 * beta;
    SubstMatrix::from_column_slice(
        4,
        4,
        &[
            -(alpha + 2.0 * beta),
            alpha,
            beta,
            beta,
            alpha,
            -(alpha + 2.0 * beta),
            beta,
            beta,
            beta,
            beta,
            -(alpha + 2.0 * beta),
            alpha,
            beta,
            beta,
            alpha,
            -(alpha + 2.0 * beta),
        ],
    )
    .div(total)
}

impl DNASubstModel {
    pub(crate) fn reset_k80_q(&mut self, params: DNASubstParams) {
        self.q = k80_q(&params);
        self.params = SubstParams::DNA(params);
    }
}

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

    pub fn optimise_parameters(&self) -> Result<(u32, DNASubstParams, f64)> {
        let epsilon = 1e-10;
        let SubstParams::DNA(mut k80_params) = self.base_model.params.clone() else {
            unreachable!();
        };
        let mut model = k80(k80_params.clone());
        let mut logl = f64::NEG_INFINITY;
        let mut new_logl = 0.0;
        let mut iters = 0;
        while (logl - new_logl).abs() > epsilon {
            logl = new_logl;
            _ = self.optimise_alpha(&mut model, &mut k80_params)?;
            new_logl = self.optimise_beta(&mut model, &mut k80_params)?;
            iters += 1;
        }
        Ok((iters, k80_params, -logl))
    }

    fn optimise_alpha(
        &self,
        model: &mut DNASubstModel,
        k80_params: &mut DNASubstParams,
    ) -> Result<f64> {
        model.reset_k80_q(k80_params.clone());
        let alpha_optimiser = K80ModelAlphaOptimiser {
            likelihood_cost: self.likelihood_cost,
            base_model: model,
        };
        let res = Executor::new(alpha_optimiser, subst_param_brent()).run()?;
        let alpha = res.state().best_param.unwrap();
        k80_params.rtc = alpha;
        k80_params.rag = alpha;
        Ok(res.state().best_cost)
    }

    fn optimise_beta(
        &self,
        model: &mut DNASubstModel,
        k80_params: &mut DNASubstParams,
    ) -> Result<f64> {
        model.reset_k80_q(k80_params.clone());
        let beta_optimiser = K80ModelBetaOptimiser {
            likelihood_cost: self.likelihood_cost,
            base_model: model,
        };
        let res = Executor::new(beta_optimiser, subst_param_brent()).run()?;
        let beta = res.state().best_param.unwrap();
        k80_params.rta = beta;
        k80_params.rtg = beta;
        k80_params.rca = beta;
        k80_params.rcg = beta;
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
        let SubstParams::DNA(mut k80_params) = model.params.clone() else {
            bail!("Incorrect substitution model parameter type.")
        };
        k80_params.rtc = *param;
        k80_params.rag = *param;
        model.reset_k80_q(k80_params);
        let mut tmp_info = SubstitutionModelInfo::new(self.likelihood_cost.info, &model)?;
        Ok(-self
            .likelihood_cost
            .compute_log_likelihood(&model, &mut tmp_info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

impl CostFunction for K80ModelBetaOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let mut model = self.base_model.clone();
        let SubstParams::DNA(mut k80_params) = model.params.clone() else {
            bail!("Incorrect substitution model parameter type.")
        };
        k80_params.rta = *param;
        k80_params.rtg = *param;
        k80_params.rca = *param;
        k80_params.rcg = *param;
        model.reset_k80_q(k80_params);
        let mut tmp_info = SubstitutionModelInfo::new(self.likelihood_cost.info, &model)?;
        Ok(-self
            .likelihood_cost
            .compute_log_likelihood(&model, &mut tmp_info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}
