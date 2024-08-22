use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::FrequencyOptimisation::{self, *};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{ModelOptimisationResult, ModelOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{SubstLikelihoodCost, SubstModel, SubstitutionModel};
use crate::Result;

pub(crate) struct SubstParamOptimiser<'a, SubstModel: SubstitutionModel> {
    pub(crate) likelihood: &'a SubstLikelihoodCost<'a, SubstModel>,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) model: &'a SubstModel,
    pub(crate) parameter: &'a [<SubstModel as SubstitutionModel>::Parameter],
}

impl<Params> CostFunction for SubstParamOptimiser<'_, SubstModel<Params>>
where
    SubstModel<Params>: SubstitutionModel + Clone,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<Self::Output> {
        let mut model = self.model.clone();
        for param_name in self.parameter {
            model.set_param(param_name, *value);
        }
        let mut likelihood = self.likelihood.clone();

        likelihood.model = &model;
        Ok(-likelihood.cost(self.info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct SubstModelOptimiser<'a, SubstModel: SubstitutionModel> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood: &'a SubstLikelihoodCost<'a, SubstModel>,
    pub(crate) info: PhyloInfo,
    pub(crate) freq_opt: FrequencyOptimisation,
}
impl<'a, Params> ModelOptimiser<'a, SubstLikelihoodCost<'a, SubstModel<Params>>, SubstModel<Params>>
    for SubstModelOptimiser<'a, SubstModel<Params>>
where
    Params: Display,
    SubstModel<Params>: SubstitutionModel + Clone,
    <SubstModel<Params> as SubstitutionModel>::ModelType: Display,
{
    fn new(
        likelihood: &'a SubstLikelihoodCost<'a, SubstModel<Params>>,
        phylo_info: &PhyloInfo,
        freq_opt: FrequencyOptimisation,
    ) -> Self {
        SubstModelOptimiser {
            epsilon: 1e-3,
            freq_opt,
            likelihood,
            info: phylo_info.clone(),
        }
    }

    fn run(self) -> Result<ModelOptimisationResult<SubstModel<Params>>> {
        let likelihood = self.likelihood.clone();
        let initial_logl = self.likelihood.cost(&self.info);
        let mut model = likelihood.model.clone();
        info!("Optimising {} parameters.", model.model_type());
        info!("Initial logl: {}.", initial_logl);

        match self.freq_opt {
            Fixed => {}
            Empirical => {
                info!("Seting stationary frequencies to empirical.");
                model.set_freqs(self.info.freqs());
            }
            Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical.");
                model.set_freqs(self.info.freqs());
            }
        }

        let mut likelihood = self.likelihood.clone();
        likelihood.model = &model;

        let mut prev_logl = f64::NEG_INFINITY;
        let mut final_logl = likelihood.cost(&self.info);
        let mut iterations = 0;

        let param_sets = model.parameter_definition();
        while (prev_logl - final_logl).abs() > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = SubstParamOptimiser {
                    likelihood: self.likelihood,
                    model: &model,
                    info: &self.info,
                    parameter: param_set,
                };
                let gss = BrentOpt::new(1e-10, 100.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| IterState::new().param(model.param(param_set.first().unwrap())))
                    .run()?;
                let logl = -res.state().best_cost;
                if logl < final_logl {
                    continue;
                }
                let value = res.state().best_param.unwrap();
                for param_id in param_set {
                    model.set_param(param_id, value);
                }
                final_logl = logl;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, final_logl
                );
                debug!("New parameters: {}\n", model.params);
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            final_logl, iterations
        );
        Ok(ModelOptimisationResult::<SubstModel<Params>> {
            model,
            initial_logl,
            final_logl,
            iterations,
        })
    }
}
