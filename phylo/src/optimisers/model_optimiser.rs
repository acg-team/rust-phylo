use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::ModelOptimisationResult;
use crate::Result;

use super::SingleValOptResult;

pub struct ModelOptimiser<'a, C: ModelSearchCost + Display> {
    pub(crate) epsilon: f64,
    pub(crate) c: &'a mut C,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<'a, C: ModelSearchCost + Display> ModelOptimiser<'a, C> {
    pub fn new(cost: &'a mut C, freq_opt: FrequencyOptimisation) -> ModelOptimiser<'a, C> {
        Self {
            epsilon: 1e-3,
            c: cost,
            freq_opt,
        }
    }

    pub fn run(mut self) -> Result<ModelOptimisationResult<'a, C>> {
        info!("Optimising the evolutionary model: {}.", self.c);

        let init_cost = self.c.cost();
        info!("Initial cost: {}.", init_cost);
        let mut curr_cost = init_cost;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        match self.freq_opt {
            FrequencyOptimisation::Empirical => {
                info!("Setting stationary frequencies to empirical.");
                self.empirical_freqs();
                curr_cost = self.c.cost();
                info!("Cost after frequency optimisation: {}.", curr_cost);
            }
            FrequencyOptimisation::Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical.");
                self.empirical_freqs();
                curr_cost = self.c.cost();
                info!("Cost after frequency optimisation: {}.", curr_cost);
            }
            FrequencyOptimisation::Fixed => {}
        }

        while (curr_cost - prev_cost) > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            let parameters = self.c.params().to_vec();
            prev_cost = curr_cost;

            for (param, start_value) in parameters.iter().enumerate() {
                debug!(
                    "Optimising parameter {:?} from value {} with cost {}",
                    param, start_value, curr_cost
                );
                let param_opt = self.opt_parameter(param, *start_value)?;
                if param_opt.final_cost < curr_cost {
                    // Parameter will have been reset by the optimiser, set it back to start value
                    self.c.set_param(param, *start_value);
                    continue;
                }
                self.c.set_param(param, param_opt.value);
                curr_cost = param_opt.final_cost;
                debug!(
                    "Optimised parameter {:?} to value {} with cost {}",
                    param, param_opt.value, curr_cost
                );
            }
            debug!("New parameters: {}\n", self.c);
        }

        debug_assert_eq!(curr_cost, self.c.cost());
        info!("Done optimising model parameters.");
        info!(
            "Final cost: {}, achieved in {} iteration(s).",
            curr_cost, iterations
        );
        Ok(ModelOptimisationResult::<C> {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            cost: self.c,
        })
    }

    fn empirical_freqs(&mut self) {
        let emp_freqs = self.c.empirical_freqs();
        self.c.set_freqs(emp_freqs);
    }

    fn opt_parameter(&mut self, param: usize, start_value: f64) -> Result<SingleValOptResult> {
        let optimiser = ParamOptimiser {
            cost: RefCell::new(self.c),
            param,
        };
        let min = f64::EPSILON;
        let max = start_value * 100.0;
        let gss = BrentOpt::new(min, max);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_value).max_iters(500))
            .run()?;
        let cost = -res.state().best_cost;
        Ok(SingleValOptResult {
            value: res.state().best_param.unwrap(),
            final_cost: cost,
        })
    }
}

pub(crate) struct ParamOptimiser<'a, C: ModelSearchCost> {
    pub(crate) cost: RefCell<&'a mut C>,
    pub(crate) param: usize,
}

impl<C: ModelSearchCost> CostFunction for ParamOptimiser<'_, C> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        self.cost.borrow_mut().set_param(self.param, *value);
        Ok(-self.cost.borrow().cost())
    }

    fn parallelize(&self) -> bool {
        true
    }
}
