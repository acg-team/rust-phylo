use crate::likelihood::{ModelSearchCost, TreeSearchCost};

pub mod blen_optimiser;
pub use blen_optimiser::*;
pub mod model_optimiser;
pub use model_optimiser::*;
pub mod topo_optimiser;
pub use topo_optimiser::*;

pub struct PhyloOptimisationResult<C: TreeSearchCost> {
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
mod topo_optimiser_tests;
