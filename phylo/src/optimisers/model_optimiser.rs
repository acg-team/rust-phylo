use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::{ModelOptimisationResult, SingleValOptResult, StopCondition};
use crate::{Result, DEFAULT_EPSILON};

pub struct ModelOptimiser<C: ModelSearchCost + Display + Clone> {
    pub(crate) stop_condition: StopCondition,
    pub(crate) c: C,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<C: ModelSearchCost + Display + Clone> ModelOptimiser<C> {
    pub fn new(cost: C, freq_opt: FrequencyOptimisation) -> Self {
        Self {
            stop_condition: StopCondition::Epsilon(DEFAULT_EPSILON),
            c: cost,
            freq_opt,
        }
    }

    pub fn with_stop_condition(
        cost: C,
        stop_condition: StopCondition,
        freq_opt: FrequencyOptimisation,
    ) -> Self {
        Self {
            stop_condition,
            c: cost,
            freq_opt,
        }
    }

    pub fn run(mut self) -> Result<ModelOptimisationResult<C>> {
        info!("Optimising the evolutionary model: {}", self.c);
        info!("Optimisation stopping condition: {}", self.stop_condition);

        let init_cost = self.c.cost();
        info!("Initial cost: {init_cost}");

        let mut curr_cost = self.optimise_frequencies();
        debug_assert!(curr_cost >= init_cost);

        // Set previous cost to negative infinity to ensure at least one iteration if frequency optimisation did not change the cost
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;
        let mut delta = curr_cost - prev_cost;
        // Store costs for each iteration, including initial cost before potential frequency optimisation
        let mut costs = vec![init_cost, curr_cost];

        while self.stop_condition.should_continue(iterations, delta) {
            iterations += 1;
            info!("Iteration: {iterations}, current cost: {curr_cost}");
            prev_cost = curr_cost;
            curr_cost = self.single_optimisation_iteration()?;
            debug!("New parameters: {}\n", self.c);
            delta = curr_cost - prev_cost;
            costs.push(curr_cost);
        }

        debug_assert_eq!(curr_cost, self.c.cost());
        info!("Done optimising model parameters");
        info!("Final cost: {curr_cost}, achieved in {iterations} iteration(s)");

        Ok(ModelOptimisationResult::<C> {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            costs,
            cost: self.c,
        })
    }

    fn optimise_frequencies(&mut self) -> f64 {
        match self.freq_opt {
            FrequencyOptimisation::Empirical => {
                info!("Setting stationary frequencies to empirical");
                self.empirical_freqs();
            }
            FrequencyOptimisation::Estimated => {
                warn!("Stationary frequency estimation not available, falling back on empirical");
                self.empirical_freqs();
            }
            FrequencyOptimisation::Fixed => {
                info!("Not optimising stationary frequencies");
            }
        }
        let cost = self.c.cost();
        info!("Cost after frequency optimisation: {cost}");
        cost
    }

    fn single_optimisation_iteration(&mut self) -> Result<f64> {
        let parameters = self.c.params().to_vec();
        let mut curr_cost = self.c.cost();

        for (param, start_value) in parameters.iter().enumerate() {
            debug!("Optimising parameter {param:?} from value {start_value} with cost {curr_cost}");
            let param_opt = self.opt_parameter(param, *start_value)?;
            if param_opt.final_cost < curr_cost {
                // Parameter will have been reset by the optimiser, set it back to start value
                self.c.set_param(param, *start_value);
                continue;
            }
            self.c.set_param(param, param_opt.value);
            curr_cost = param_opt.final_cost;
            debug!(
                "Optimised parameter {param:?} to value {} with cost {curr_cost}",
                param_opt.value
            );
        }
        Ok(curr_cost)
    }

    fn empirical_freqs(&mut self) {
        let emp_freqs = self.c.empirical_freqs();
        self.c.set_freqs(emp_freqs);
    }

    fn opt_parameter(&self, param: usize, start_value: f64) -> Result<SingleValOptResult> {
        let optimiser = ParamOptimiser {
            cost: RefCell::new(self.c.clone()),
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

pub(crate) struct ParamOptimiser<C: ModelSearchCost> {
    pub(crate) cost: RefCell<C>,
    pub(crate) param: usize,
}

impl<C: ModelSearchCost> CostFunction for ParamOptimiser<C> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        let value = if value.is_nan() || value.is_sign_negative() {
            0.0
        } else {
            *value
        };
        self.cost.borrow_mut().set_param(self.param, value);
        Ok(-self.cost.borrow().cost())
    }

    fn parallelize(&self) -> bool {
        true
    }
}
