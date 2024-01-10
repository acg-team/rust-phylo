use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::substitution_models::dna_models::{gtr, DNASubstParams, ParamEnum};
use crate::substitution_models::SubstitutionLikelihoodCost;
use crate::Result;

pub(crate) struct DNAParamOptimiser<'a> {
    pub(crate) likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    pub(crate) params: DNASubstParams,
    pub(crate) parameter: &'a [ParamEnum],
}

impl CostFunction for DNAParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut gtr_params = self.params.clone();
        for param_name in self.parameter {
            gtr_params.set_value(param_name, *value);
        }
        let model = gtr(gtr_params);
        Ok(-self.likelihood_cost.compute_log_likelihood(&model).0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct DNAModelOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
}

impl<'a> DNAModelOptimiser<'a> {
    pub fn new(likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>) -> Self {
        DNAModelOptimiser {
            epsilon: 1e-3,
            likelihood_cost,
        }
    }

    pub fn optimise_k80_parameters(
        &self,
        start_values: &DNASubstParams,
    ) -> Result<(u32, DNASubstParams, f64)> {
        println!("Optimising K80 parameters.");
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
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub fn optimise_hky_parameters(
        &self,
        start_values: &DNASubstParams,
    ) -> Result<(u32, DNASubstParams, f64)> {
        println!("Optimising HKY parameters.");
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
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub fn optimise_tn93_parameters(
        &self,
        start_values: &DNASubstParams,
    ) -> Result<(u32, DNASubstParams, f64)> {
        println!("Optimising HKY parameters.");
        let param_sets = [
            ("alpha1", vec![ParamEnum::Rtc]),
            ("alpha2", vec![ParamEnum::Rag]),
            (
                "beta",
                vec![
                    ParamEnum::Rta,
                    ParamEnum::Rtg,
                    ParamEnum::Rca,
                    ParamEnum::Rcg,
                ],
            ),
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub fn optimise_gtr_parameters(
        &self,
        start_values: &DNASubstParams,
    ) -> Result<(u32, DNASubstParams, f64)> {
        println!("Optimising GTR parameters.");
        let param_sets = [
            ("rca", vec![ParamEnum::Rca]),
            ("rcg", vec![ParamEnum::Rcg]),
            ("rta", vec![ParamEnum::Rta]),
            ("rtc", vec![ParamEnum::Rtc]),
            ("rtg", vec![ParamEnum::Rtg]),
        ];
        self.optimise_parameters(start_values, &param_sets)
    }

    pub(crate) fn optimise_parameters(
        &self,
        start_values: &DNASubstParams,
        param_sets: &[(&str, Vec<ParamEnum>)],
    ) -> Result<(u32, DNASubstParams, f64)> {
        let mut opt_logl = self
            .likelihood_cost
            .compute_log_likelihood(&gtr(start_values.clone()))
            .0;
        info!("Initial logl: {}.", opt_logl);
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;
        let mut opt_params = start_values.clone();
        while (prev_logl - opt_logl).abs() > self.epsilon {
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = DNAParamOptimiser {
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
                debug!("New parameters: {}\n", opt_params.print_as_gtr());
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
