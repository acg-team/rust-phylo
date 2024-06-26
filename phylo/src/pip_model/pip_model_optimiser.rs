use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::pip_model::PIPLikelihoodCost;
use crate::pip_model::PIPModel;
use crate::substitution_models::dna_models::{gtr, ParamEnum, NUCLEOTIDE_INDEX};
use crate::Result;

use super::PIPDNAParams;

pub(crate) struct PIPDNAParamOptimiser<'a> {
    pub(crate) likelihood_cost: &'a PIPLikelihoodCost<'a, 4>,
    pub(crate) params: PIPDNAParams,
    pub(crate) parameter: &'a [ParamEnum],
}

impl CostFunction for PIPDNAParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut params = self.params.clone();
        for param_name in self.parameter {
            params.set_value(param_name, *value);
        }

        let subst_model = gtr(params.subst_params.clone());
        let index = *NUCLEOTIDE_INDEX;
        let model = PIPModel::make_pip(index, subst_model, params.mu, params.lambda);
        Ok(-self.likelihood_cost.compute_log_likelihood(&model).0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct PIPDNAModelOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood_cost: &'a PIPLikelihoodCost<'a, 4>,
}

impl<'a> PIPDNAModelOptimiser<'a> {
    pub fn new(likelihood_cost: &'a PIPLikelihoodCost<'a, 4>) -> Self {
        PIPDNAModelOptimiser {
            epsilon: 1e-3,
            likelihood_cost,
        }
    }

    pub fn optimise_gtr_parameters(
        &self,
        start_values: &PIPDNAParams,
    ) -> Result<(u32, PIPDNAParams, f64)> {
        info!("Optimising PIP with GTR parameters.");
        let param_sets = [
            ("rca", vec![ParamEnum::Rca]),
            ("rcg", vec![ParamEnum::Rcg]),
            ("rta", vec![ParamEnum::Rta]),
            ("rtc", vec![ParamEnum::Rtc]),
            ("rtg", vec![ParamEnum::Rtg]),
            ("mu", vec![ParamEnum::Mu]),
            ("lambda", vec![ParamEnum::Lambda]),
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub fn optimise_jc69_parameters(
        &self,
        start_values: &PIPDNAParams,
    ) -> Result<(u32, PIPDNAParams, f64)> {
        info!("Optimising PIP with JC69 parameters.");
        let param_sets = [
            ("mu", vec![ParamEnum::Mu]),
            ("lambda", vec![ParamEnum::Lambda]),
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub fn optimise_hky_parameters(
        &self,
        start_values: &PIPDNAParams,
    ) -> Result<(u32, PIPDNAParams, f64)> {
        info!("Optimising HKY parameters.");
        let param_sets = [
            ("alpha", vec![ParamEnum::Rtc, ParamEnum::Rag]),
            (
                "beta",
                vec![
                    ParamEnum::Rta,
                    ParamEnum::Rtg,
                    ParamEnum::Rca,
                    ParamEnum::Rcg,
                ],
            ),
            ("mu", vec![ParamEnum::Mu]),
            ("lambda", vec![ParamEnum::Lambda]),
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub(crate) fn optimise_parameters(
        &self,
        start_values: &PIPDNAParams,
        param_sets: &[(&str, Vec<ParamEnum>)],
    ) -> Result<(u32, PIPDNAParams, f64)> {
        info!("Optimising PIP parameters.");
        let pip = PIPModel::<4>::make_pip(
            *NUCLEOTIDE_INDEX,
            gtr(start_values.subst_params.clone()),
            start_values.mu,
            start_values.lambda,
        );
        let mut opt_logl = self.likelihood_cost.compute_log_likelihood(&pip).0;
        info!("Initial logl: {}.", opt_logl);
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;
        let mut opt_params = start_values.clone();
        while (prev_logl - opt_logl).abs() > self.epsilon {
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = PIPDNAParamOptimiser {
                    likelihood_cost: self.likelihood_cost,
                    params: opt_params.clone(),
                    parameter: param_set,
                };
                let gss = BrentOpt::new(1e-10, 100.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| {
                        IterState::new().param(opt_params.get_value(param_set.first().unwrap()))
                    })
                    .run()?;
                let value = res.state().best_param.unwrap();
                for param_id in param_set {
                    opt_params.set_value(param_id, value);
                }
                opt_logl = -res.state().best_cost;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, opt_logl
                );
                debug!("New parameters: {}\n", opt_params);
            }
            iters += 1;
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok((iters, opt_params, opt_logl))
    }
}
