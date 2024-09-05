use std::fmt::{Debug, Display};

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{EvoModelOptimisationResult, EvoModelOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::Result;

pub(crate) struct ParamOptimiser<'a, EM: EvoModel + PhyloCostFunction> {
    pub(crate) model: &'a EM,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) parameter: &'a EM::Parameter,
}

impl<'a, EM: EvoModel + PhyloCostFunction + Clone> CostFunction for ParamOptimiser<'a, EM> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        let mut model = self.model.clone();
        model.set_param(self.parameter, *value);
        Ok(-model.cost(self.info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct ModelOptimiser<'a, EM: EvoModel + PhyloCostFunction + Clone> {
    pub(crate) epsilon: f64,
    pub(crate) model: &'a EM,
    pub(crate) info: PhyloInfo,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<'a, EM: EvoModel + PhyloCostFunction + Clone + Display> EvoModelOptimiser<'a, EM>
    for ModelOptimiser<'a, EM>
where
    EM::Parameter: Debug,
    EM::ModelType: Display,
{
    fn new(model: &'a EM, info: &PhyloInfo, freq_opt: FrequencyOptimisation) -> Self {
        Self {
            epsilon: 1e-3,
            model,
            info: info.clone(),
            freq_opt,
        }
    }

    fn run(self) -> Result<EvoModelOptimisationResult<EM>> {
        let mut model = self.model.clone();
        let initial_logl = model.cost(&self.info);
        info!("Optimising {} parameters.", model.description());
        info!("Initial logl: {}.", initial_logl);

        self.opt_frequencies(&mut model);

        let mut prev_logl = f64::NEG_INFINITY;
        let mut final_logl = model.cost(&self.info);
        let mut iterations = 0;

        let parameters = model.model_parameters();
        while (prev_logl - final_logl).abs() > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for param_name in parameters.iter() {
                let (value, logl) = self.opt_parameter(&model, param_name)?;
                if logl < final_logl {
                    continue;
                }
                model.set_param(param_name, value);
                final_logl = logl;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param_name, value, final_logl
                );
            }
            debug!("New parameters: {}\n", model);
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            final_logl, iterations
        );
        Ok(EvoModelOptimisationResult::<EM> {
            model,
            initial_logl,
            final_logl,
            iterations,
        })
    }
}

impl<'a, EM: EvoModel + PhyloCostFunction + Clone + Display> ModelOptimiser<'a, EM> {
    fn opt_frequencies(&self, model: &mut EM) {
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
    }

    fn opt_parameter(
        &self,
        model: &EM,
        parameter: &<EM as EvoModel>::Parameter,
    ) -> Result<(f64, f64)> {
        let optimiser = ParamOptimiser {
            model,
            info: &self.info,
            parameter,
        };
        let gss = BrentOpt::new(1e-10, 100.0);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(model.param(parameter)))
            .run()?;
        let logl = -res.state().best_cost;
        Ok((res.state().best_param.unwrap(), logl))
    }
}
