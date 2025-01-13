use crate::likelihood::{ModelSearchCost, TreeSearchCost};

pub mod blen_optimiser;
pub use blen_optimiser::*;
pub mod model_optimiser;
pub use model_optimiser::*;
pub mod topo_optimiser;
pub use topo_optimiser::*;

pub struct PhyloOptimisationResult<C: TreeSearchCost + Clone> {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub cost: C,
}

pub struct ModelOptimisationResult<C: ModelSearchCost + Clone> {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub cost: C,
}

#[cfg(test)]
mod blen_optimiser_tests;
#[cfg(test)]
mod model_optimiser_tests;
#[cfg(test)]
mod topo_optimiser_tests;
