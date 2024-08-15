use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::{
    DNAModelType::*,
    EvoModelParams,
    FrequencyOptimisation::{self, *},
};
use crate::likelihood::LikelihoodCostFunction;
use crate::substitution_models::dna_models::{DNAParameter, DNASubstModel, DNASubstParams};
use crate::substitution_models::{SubstitutionLikelihoodCost, SubstitutionModel};
use crate::Result;

pub(crate) struct DNAParamOptimiser<'a> {
    pub(crate) likelihood_cost: &'a SubstitutionLikelihoodCost<'a, DNASubstModel>,
    pub(crate) model: &'a DNASubstModel,
    pub(crate) parameter: &'a [DNAParameter],
}

impl CostFunction for DNAParamOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut gtr_params = self.model.params.clone();
        for param_name in self.parameter {
            gtr_params.set_value(param_name, *value);
        }
        let mut likelihood_cost = self.likelihood_cost.clone();
        let model = DNASubstModel::create(&gtr_params);
        likelihood_cost.model = &model;
        Ok(-likelihood_cost.compute_log_likelihood().0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct DNAModelOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood_cost: &'a SubstitutionLikelihoodCost<'a, DNASubstModel>,
}

impl<'a> DNAModelOptimiser<'a> {
    pub fn new(likelihood_cost: &'a SubstitutionLikelihoodCost<'a, DNASubstModel>) -> Self {
        DNAModelOptimiser {
            epsilon: 1e-3,
            likelihood_cost,
        }
    }

    fn set_empirical_frequencies(&self, start_params: &DNASubstParams) -> DNASubstParams {
        let mut start_params = start_params.clone();
        start_params.set_freqs(self.likelihood_cost.empirical_frequencies());
        info!("Set stationary frequencies to empirical.");
        start_params
    }

    pub fn optimise_parameters(
        &self,
        optimise_freqs: FrequencyOptimisation,
    ) -> Result<(u32, DNASubstParams, f64)> {
        let start_params = self.likelihood_cost.model.params.clone();
        let model_type = start_params.model_type;
        info!("Optimising {} parameters.", model_type);
        let param_sets = DNASubstParams::parameter_definition(&model_type);
        let start_params = match model_type {
            JC69 | K80 => start_params,
            _ => {
                match optimise_freqs {
                    Fixed => start_params,
                    Empirical => self.set_empirical_frequencies(&start_params),
                    Estimated => {
                        warn!("Stationary frequency estimation not available, falling back on empirical.");
                        self.set_empirical_frequencies(&start_params)
                    }
                }
            }
        };
        self.run_parameter_brent(&start_params, param_sets)
    }

    fn run_parameter_brent(
        &self,
        start_params: &DNASubstParams,
        param_sets: Vec<(&str, Vec<DNAParameter>)>,
    ) -> Result<(u32, DNASubstParams, f64)> {
        let mut prev_logl = f64::NEG_INFINITY;
        let mut opt_logl = self.likelihood_cost.compute_log_likelihood().0;
        info!("Initial logl: {}.", opt_logl);
        let mut opt_params = start_params.clone();
        let mut model = DNASubstModel::create(&opt_params);
        let mut iters = 0;

        while (prev_logl - opt_logl).abs() > self.epsilon {
            iters += 1;
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = DNAParamOptimiser {
                    likelihood_cost: self.likelihood_cost,
                    model: &model,
                    parameter: param_set,
                };
                let gss = BrentOpt::new(1e-10, 100.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| {
                        IterState::new().param(opt_params.value(param_set.first().unwrap()))
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
                model = DNASubstModel::create(&opt_params);
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok((iters, opt_params, opt_logl))
    }
}
