use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::EvoModel;
use crate::evolutionary_models::FrequencyOptimisation::{self, *};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{EvoModelOptimisationResult, EvoModelOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{SubstModel, SubstitutionModel};
use crate::Result;

pub(crate) struct SubstParamOptimiser<'a, SM: SubstitutionModel + EvoModel + PhyloCostFunction> {
    pub(crate) info: &'a PhyloInfo,
    pub(crate) model: &'a SM,
    pub(crate) parameter: &'a [<SM as EvoModel>::Parameter],
}

impl<Params> CostFunction for SubstParamOptimiser<'_, SubstModel<Params>>
where
    SubstModel<Params>: SubstitutionModel + EvoModel + PhyloCostFunction + Clone,
{
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<Self::Output> {
        let mut model = self.model.clone();
        for param_name in self.parameter {
            EvoModel::set_param(&mut model, param_name, *value);
        }
        Ok(-model.cost(self.info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct SubstModelOptimiser<'a, SM: SubstitutionModel + EvoModel + PhyloCostFunction> {
    pub(crate) epsilon: f64,
    pub(crate) model: &'a SM,
    pub(crate) info: PhyloInfo,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<'a, Params> EvoModelOptimiser<'a, SubstModel<Params>>
    for SubstModelOptimiser<'a, SubstModel<Params>>
where
    Params: Display,
    SubstModel<Params>: SubstitutionModel + Clone,
    <SubstModel<Params> as SubstitutionModel>::ModelType: Display,
{
    fn new(
        model: &'a SubstModel<Params>,
        phylo_info: &PhyloInfo,
        freq_opt: FrequencyOptimisation,
    ) -> Self {
        SubstModelOptimiser {
            epsilon: 1e-3,
            freq_opt,
            model,
            info: phylo_info.clone(),
        }
    }

    fn run(self) -> Result<EvoModelOptimisationResult<SubstModel<Params>>> {
        let mut model = self.model.clone();
        let initial_logl = model.cost(&self.info);
        info!(
            "Optimising {} parameters.",
            SubstitutionModel::model_type(&model)
        );
        info!("Initial logl: {}.", initial_logl);

        match self.freq_opt {
            Fixed => {}
            Empirical => {
                info!("Seting stationary frequencies to empirical.");
                EvoModel::set_freqs(&mut model, self.info.freqs());
            }
            Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical.");
                EvoModel::set_freqs(&mut model, self.info.freqs());
            }
        }

        let mut prev_logl = f64::NEG_INFINITY;
        let mut final_logl = model.cost(&self.info);
        let mut iterations = 0;

        let param_sets = EvoModel::parameter_definition(&model);
        while (prev_logl - final_logl).abs() > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for (param_name, param_set) in param_sets.iter() {
                let optimiser = SubstParamOptimiser {
                    model: &model,
                    info: &self.info,
                    parameter: param_set,
                };
                let gss = BrentOpt::new(1e-10, 100.0);
                let res = Executor::new(optimiser, gss)
                    .configure(|_| {
                        IterState::new().param(EvoModel::param(&model, param_set.first().unwrap()))
                    })
                    .run()?;
                let logl = -res.state().best_cost;
                if logl < final_logl {
                    continue;
                }
                let value = res.state().best_param.unwrap();
                for param_id in param_set {
                    EvoModel::set_param(&mut model, param_id, value);
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
        Ok(EvoModelOptimisationResult::<SubstModel<Params>> {
            model,
            initial_logl,
            final_logl,
            iterations,
        })
    }
}
