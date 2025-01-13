use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::evolutionary_models::FrequencyOptimisation;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::ModelOptimisationResult;
use crate::Result;

pub struct ModelOptimiser<C: ModelSearchCost + Display + Clone> {
    pub(crate) epsilon: f64,
    pub(crate) c: RefCell<C>,
    pub(crate) freq_opt: FrequencyOptimisation,
}

impl<C: ModelSearchCost + Display + Clone> ModelOptimiser<C> {
    pub fn new(cost: C, freq_opt: FrequencyOptimisation) -> Self {
        Self {
            epsilon: 1e-3,
            c: RefCell::new(cost),
            freq_opt,
        }
    }

    pub fn run(self) -> Result<ModelOptimisationResult<C>> {
        let initial_logl = self.c.borrow().cost();
        info!("Optimising {}.", self.c.borrow());
        info!("Initial logl: {}.", initial_logl);

        self.opt_frequencies();

        let mut prev_logl = f64::NEG_INFINITY;
        let mut final_logl = self.c.borrow().cost();
        info!("Initial logl after frequency optimisation: {}.", final_logl);

        let mut iterations = 0;

        let parameters = self.c.borrow().params().to_vec();
        while final_logl - prev_logl > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for (param, value) in parameters.iter().enumerate() {
                let (value, logl) = self.opt_parameter(param, *value)?;
                if logl < final_logl {
                    continue;
                }
                self.c.borrow_mut().set_param(param, value);
                final_logl = logl;
                debug!(
                    "Optimised parameter {:?} to value {} with logl {}",
                    param, value, final_logl
                );
            }
            debug!("New parameters: {}\n", self.c.borrow());
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            final_logl, iterations
        );
        Ok(ModelOptimisationResult::<C> {
            cost: self.c.into_inner(),
            initial_logl,
            final_logl,
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
        let gss = BrentOpt::new(1e-10, 100.0);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_value))
            .run()?;
        let logl = -res.state().best_cost;
        Ok((res.state().best_param.unwrap(), logl))
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
