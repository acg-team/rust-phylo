use std::num::NonZeroUsize;

use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::DEFAULT_EPSILON;

pub mod blen_optimiser;
pub use blen_optimiser::*;
pub mod model_optimiser;
pub use model_optimiser::*;
pub mod topo_optimiser;
pub use topo_optimiser::*;
pub mod spr_optimiser;
pub use spr_optimiser::*;
pub mod nni_optimiser;
pub use nni_optimiser::*;
pub mod move_optimiser;
pub use move_optimiser::*;

#[derive(Debug, Clone, Copy)]
pub enum StopCondition {
    Epsilon(f64),
    FixedIter(NonZeroUsize),
    MaxIterEpsilon(NonZeroUsize, f64),
    // NOTE: use of `fn(..) -> ..` disallows closures that capture any
    // surrounding variables, for that we would need to allow Boxed Fn
    // trait objects (or introduce a generic parameter which might get tedious)
    Custom(fn(usize, f64) -> bool),
}

impl std::fmt::Display for StopCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopCondition::Epsilon(e) => write!(f, "delta cost < {e}"),
            StopCondition::FixedIter(n) => write!(f, "fixed number of iterations = {}", n.get()),
            StopCondition::MaxIterEpsilon(n, e) => {
                write!(f, "delta cost < {e}, max iterations = {}", n.get())
            }
            StopCondition::Custom(fun) => write!(f, "custom predicate function: {fun:?}"),
        }
    }
}

impl StopCondition {
    /// Validates whether the optimisation should continue based on the current iteration
    /// number and the change in cost (delta) since the last iteration.
    /// Returns `true` if the optimisation should continue, `false` otherwise
    fn should_continue(&self, iteration: usize, delta: f64) -> bool {
        match *self {
            StopCondition::Epsilon(min_delta) => delta > min_delta,
            StopCondition::FixedIter(max) => max.get() > iteration,
            StopCondition::MaxIterEpsilon(max, min_delta) => {
                max.get() > iteration && delta > min_delta
            }
            StopCondition::Custom(pred) => pred(iteration, delta),
        }
    }

    pub fn epsilon(epsilon: f64) -> Self {
        Self::Epsilon(epsilon)
    }

    pub fn fixed_iter(num: NonZeroUsize) -> Self {
        Self::FixedIter(num)
    }

    pub fn max_iter_epsilon(num: NonZeroUsize, epsilon: f64) -> Self {
        Self::MaxIterEpsilon(num, epsilon)
    }

    pub fn max_iter(num: NonZeroUsize) -> Self {
        Self::MaxIterEpsilon(num, DEFAULT_EPSILON)
    }

    pub fn custom(pred: fn(usize, f64) -> bool) -> Self {
        Self::Custom(pred)
    }
}

// Struct for any single value optimisation result, e.g. branch length or evolutionary model parameter value
pub struct SingleValOptResult {
    // final cost after optimisation
    pub final_cost: f64,
    // value of the parameter after optimisation
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct PhyloOptimisationResult<C: TreeSearchCost> {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    #[allow(dead_code)]
    pub(crate) costs: Vec<f64>,
    pub cost: C,
}

#[derive(Clone, Debug)]
pub struct ModelOptimisationResult<C: ModelSearchCost> {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub cost: C,
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod blen_optimiser_tests;
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod model_optimiser_tests;
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod topo_optimiser_tests;

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use std::num::NonZeroUsize;

    use rstest::rstest;

    use super::StopCondition;

    #[rstest]
    #[case(0, 0.1, false)]
    #[case(1, 0.02, false)]
    #[case(2, 0.005, true)]
    #[case(3, 0.0, true)]
    fn predicate_epsilon(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        let pred = StopCondition::Epsilon(0.01);
        assert_eq!(pred.should_continue(iters, delta), !expected);
    }

    #[rstest]
    #[case(0, 0.00000001, false)]
    #[case(1, 0.02, false)]
    #[case(2, 0.000005, false)]
    #[case(3, 0.0, true)]
    #[case(3, 1.0, true)]
    #[case(5, 1e-10, true)]
    fn predicate_fixed_iter(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        let pred = StopCondition::FixedIter(NonZeroUsize::new(3).unwrap());
        assert_eq!(pred.should_continue(iters, delta), !expected);
    }

    #[rstest]
    #[case(0, 0.00000001, true)]
    #[case(1, 0.02, false)]
    #[case(2, 0.000005, true)]
    #[case(3, 0.0, true)]
    #[case(3, 1.0, true)]
    #[case(5, 1e-10, true)]
    #[case(5, 1e10, true)]
    fn predicate_max_iter(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        let pred = StopCondition::MaxIterEpsilon(NonZeroUsize::new(3).unwrap(), 0.01);
        assert_eq!(pred.should_continue(iters, delta), !expected);
    }

    #[rstest]
    #[case(0, 0.00000001, false)]
    #[case(1, 0.02, false)]
    #[case(2, 0.000005, true)]
    #[case(3, 0.0, true)]
    #[case(3, 1.0, false)]
    #[case(5, 1e-10, true)]
    #[case(5, 1e10, false)]
    fn predicate_custom(#[case] iters: usize, #[case] delta: f64, #[case] expected: bool) {
        fn custom(i: usize, d: f64) -> bool {
            !(i >= 2 && d < 0.01)
        }
        let pred = StopCondition::Custom(custom);
        assert_eq!(pred.should_continue(iters, delta), !expected);
    }
}
