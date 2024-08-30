use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::likelihood::PhyloCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::Result;

pub mod blen_optimiser;
pub use blen_optimiser::*;
pub mod model_optimiser;
pub use model_optimiser::*;
pub mod topo_optimiser;
pub use topo_optimiser::*;

pub struct PhyloOptimisationResult {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub i: PhyloInfo,
}

pub trait PhyloOptimiser<'a, EM: PhyloCostFunction> {
    fn new(cost: &'a EM, info: &PhyloInfo) -> Self;
    fn run(self) -> Result<PhyloOptimisationResult>;
}

pub struct EvoModelOptimisationResult<EM: EvoModel> {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub model: EM,
}

pub trait EvoModelOptimiser<'a, EM: EvoModel + PhyloCostFunction> {
    fn new(model: &'a EM, info: &PhyloInfo, frequencies: FrequencyOptimisation) -> Self;
    fn run(self) -> Result<EvoModelOptimisationResult<EM>>;
}

#[cfg(test)]
mod blen_optimiser_tests;
#[cfg(test)]
mod model_optimiser_tests;
