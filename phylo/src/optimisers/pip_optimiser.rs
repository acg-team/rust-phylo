use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::evolutionary_models::{EvoModel, EvoModelParams, FrequencyOptimisation};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{ModelOptimisationResult, ModelOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::pip_model::{PIPCost, PIPModel, PIPParams};
use crate::substitution_models::SubstitutionModel;
use crate::Result;

pub(crate) struct PIPParamOptimiser<'a, SubstModel: SubstitutionModel>
where
    PIPParams<SubstModel>: EvoModelParams,
{
    pub(crate) likelihood: &'a PIPCost<'a, SubstModel>,
    pub(crate) model: &'a PIPModel<SubstModel>,
    pub(crate) phylo_info: &'a PhyloInfo,
    pub(crate) parameter: &'a [<PIPParams<SubstModel> as EvoModelParams>::Parameter],
}

impl<'a, SubstModel: SubstitutionModel + Clone> CostFunction for PIPParamOptimiser<'a, SubstModel>
where
    SubstModel::ModelType: Clone,
    SubstModel::Params: Clone,
    PIPParams<SubstModel>: EvoModelParams,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        let mut params = self.model.params().clone();
        for param_name in self.parameter {
            params.set_value(param_name, *value);
        }
        let mut likelihood = self.likelihood.clone();

        let model = PIPModel::<SubstModel>::create(&params);

        likelihood.model = &model;
        Ok(-likelihood.cost(self.phylo_info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct PIPOptimiser<'a, SubstModel: SubstitutionModel> {
    pub(crate) epsilon: f64,
    pub(crate) likelihood: &'a PIPCost<'a, SubstModel>,
    pub(crate) info: PhyloInfo,
}

impl<'a, SubstModel: SubstitutionModel + Clone>
    ModelOptimiser<'a, PIPCost<'a, SubstModel>, PIPModel<SubstModel>>
    for PIPOptimiser<'a, SubstModel>
where
    SubstModel::ModelType: Clone + Display,
    SubstModel::Params: Clone,
    PIPParams<SubstModel>: EvoModelParams + Display,
{
    fn new(
        likelihood: &'a PIPCost<'a, SubstModel>,
        phylo_info: &PhyloInfo,
        _: FrequencyOptimisation,
    ) -> Self {
        Self {
            epsilon: 1e-3,
            likelihood,
            info: phylo_info.clone(),
        }
    }

    fn run(self) -> Result<ModelOptimisationResult<PIPModel<SubstModel>>> {
        let mut opt_params = self.likelihood.model.params.clone();
        let model_type = &opt_params.model_type;
        info!("Optimising PIP with {} parameters.", model_type);

        let param_sets = self.likelihood.model.params.parameter_definition();

        let initial_logl = self.likelihood.cost(&self.info);
        info!("Initial logl: {}.", initial_logl);
        let mut final_logl = initial_logl;
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut model = PIPModel::create(&opt_params);

        while (prev_logl - final_logl).abs() > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = PIPParamOptimiser {
                    likelihood: self.likelihood,
                    model: &model,
                    parameter: param_set,
                    phylo_info: &self.info,
                };
                let gss = BrentOpt::new(1e-10, 20.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| {
                        IterState::new().param(opt_params.value(param_set.first().unwrap()))
                    })
                    .run()?;
                let value = res.state().best_param.unwrap();
                for param_id in param_set {
                    opt_params.set_value(param_id, value);
                }
                final_logl = -res.state().best_cost;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, final_logl
                );
                debug_assert!(final_logl > initial_logl);
                debug_assert!(20.0 - value > 1e-10);
                debug!("New parameters: {}\n", opt_params);
                model = PIPModel::create(&opt_params);
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            final_logl, iterations
        );
        Ok(ModelOptimisationResult::<PIPModel<SubstModel>> {
            model,
            initial_logl,
            final_logl,
            iterations,
        })
    }
}
