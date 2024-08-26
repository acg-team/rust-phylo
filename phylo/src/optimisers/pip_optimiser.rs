use std::fmt::{Debug, Display};

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{ModelOptimisationResult, ModelOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::pip_model::{PIPCost, PIPModel, PIPParams};
use crate::substitution_models::SubstitutionModel;
use crate::Result;

pub(crate) struct PIPParamOptimiser<'a, SM: SubstitutionModel>
where
    PIPModel<SM>: EvoModel,
    SM: Clone,
    SM::ModelType: Clone,
{
    pub(crate) likelihood: &'a PIPCost<'a, SM>,
    pub(crate) model: &'a PIPModel<SM>,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) parameter: &'a [<PIPModel<SM> as EvoModel>::Parameter],
}

impl<'a, SM: SubstitutionModel + Clone> CostFunction for PIPParamOptimiser<'a, SM>
where
    SM::ModelType: Clone,
    PIPModel<SM>: EvoModel + Clone,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
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

pub struct PIPOptimiser<'a, SM: SubstitutionModel>
where
    PIPModel<SM>: EvoModel,
    SM: Clone,
    SM::ModelType: Clone,
{
    pub(crate) epsilon: f64,
    pub(crate) likelihood: &'a PIPCost<'a, SM>,
    pub(crate) info: PhyloInfo,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<'a, SM: SubstitutionModel + Clone> ModelOptimiser<'a, PIPCost<'a, SM>, PIPModel<SM>>
    for PIPOptimiser<'a, SM>
where
    SM::ModelType: Clone + Display,
    PIPParams<SM>: Display,
    PIPModel<SM>: EvoModel + Clone,
    <PIPModel<SM> as EvoModel>::Parameter: Debug,
{
    fn new(
        likelihood: &'a PIPCost<'a, SM>,
        phylo_info: &PhyloInfo,
        freq_opt: FrequencyOptimisation,
    ) -> Self {
        Self {
            epsilon: 1e-3,
            likelihood,
            info: phylo_info.clone(),
            freq_opt,
        }
    }

    fn run(self) -> Result<ModelOptimisationResult<PIPModel<SM>>> {
        let mut likelihood = self.likelihood.clone();
        let initial_logl = self.likelihood.cost(&self.info);
        let mut model = likelihood.model.clone();
        info!(
            "Optimising PIP with {} parameters.",
            model.params.subst_model.model_type()
        );
        info!("Initial logl: {}.", initial_logl);

        match self.freq_opt {
            FrequencyOptimisation::Fixed => {}
            FrequencyOptimisation::Empirical => {
                info!("Seting stationary frequencies to empirical.");
                model.set_freqs(self.info.freqs());
            }
            FrequencyOptimisation::Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical.");
                model.set_freqs(self.info.freqs());
            }
        }

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
                let optimiser = PIPParamOptimiser {
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
        Ok(ModelOptimisationResult::<PIPModel<SM>> {
            model,
            initial_logl,
            final_logl,
            iterations,
        })
    }
}
