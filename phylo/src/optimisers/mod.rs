use std::marker::PhantomData;

use crate::likelihood::{ModelSearchCost, TreeSearchCost};

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
pub mod tree_mover;
pub use tree_mover::*;

// Struct for any single value optimisation result, e.g. branch length or evolutionary model parameter value
pub struct SingleValOptResult {
    // final cost after optimisation
    pub final_cost: f64,
    // value of the parameter after optimisation
    pub value: f64,
}

pub struct PhyloOptimisationResult<C: TreeSearchCost<TM>, TM: TreeMover> {
    phantom: PhantomData<TM>,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub cost: C,
}

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
mod nni_optimiser_tests;
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod topo_optimiser_tests;
