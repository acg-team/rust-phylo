use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::ModelOptimisationResult;
use crate::Result;

pub struct ModelOptimiser<C: ModelSearchCost + Display> {
    pub(crate) epsilon: f64,
    pub(crate) c: RefCell<C>,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<C: ModelSearchCost + Display> ModelOptimiser<C> {
    pub fn new(cost: C, freq_opt: FrequencyOptimisation) -> Self {
        Self {
            epsilon: 1e-3,
            c: RefCell::new(cost),
            freq_opt,
        }
    }

    pub fn run(self) -> Result<ModelOptimisationResult<C>> {
        let initial_cost = self.c.borrow().cost();
        info!("Optimising {}.", self.c.borrow());
        info!("Initial cost: {}.", initial_cost);

        self.opt_frequencies();

        let mut prev_cost = f64::NEG_INFINITY;
        let mut curr_cost = self.c.borrow().cost();
        info!("Cost after frequency optimisation: {}.", curr_cost);

        let mut iterations = 0;

        while curr_cost - prev_cost > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            let parameters = self.c.borrow().params().to_vec();
            prev_cost = curr_cost;
            for (param, start_value) in parameters.iter().enumerate() {
                debug!(
                    "Optimising parameter {:?} from value {} with cost {}",
                    param, start_value, curr_cost
                );
                let (value, best_param_cost) = self.opt_parameter(param, *start_value)?;
                if best_param_cost < curr_cost {
                    // Parameter will have been reset by the optimiser, set it back to start value
                    self.c.borrow_mut().set_param(param, *start_value);
                    continue;
                }
                self.c.borrow_mut().set_param(param, value);
                curr_cost = best_param_cost;
                debug!(
                    "Optimised parameter {:?} to value {} with cost {}",
                    param, value, curr_cost
                );
            }
            debug!("New parameters: {}\n", self.c.borrow());
        }
        info!(
            "Final cost: {}, achieved in {} iteration(s).",
            curr_cost, iterations
        );
        Ok(ModelOptimisationResult::<C> {
            cost: self.c.into_inner(),
            initial_cost,
            final_cost: curr_cost,
            iterations,
        })
    }

    fn opt_frequencies(&self) {
        match self.freq_opt {
            FrequencyOptimisation::Fixed => {}
            FrequencyOptimisation::Empirical => {
                info!("Setting stationary frequencies to empirical.");
                let emp_freqs = self.c.borrow().empirical_freqs();
                self.c.borrow_mut().set_freqs(emp_freqs);
            }
            FrequencyOptimisation::Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical.");
                let emp_freqs = self.c.borrow().empirical_freqs();
                self.c.borrow_mut().set_freqs(emp_freqs);
            }
        }
    }

    fn opt_parameter(&self, param: usize, start_value: f64) -> Result<(f64, f64)> {
        let optimiser = ParamOptimiser {
            cost: &self.c,
            param,
        };
        let min = f64::EPSILON;
        let max = start_value * 100.0;
        let gss = BrentOpt::new(min, max);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_value).max_iters(500))
            .run()?;
        let cost = -res.state().best_cost;
        Ok((res.state().best_param.unwrap(), cost))
    }
}

pub(crate) struct ParamOptimiser<'a, C: ModelSearchCost> {
    pub(crate) cost: &'a RefCell<C>,
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
