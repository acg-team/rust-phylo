use crate::alignment::Alignment;
use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::likelihood::PhyloCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::tree::Tree;
use crate::Result;

pub mod blen_optimiser;
pub use blen_optimiser::*;
pub mod subst_model_optimiser;
pub use subst_model_optimiser::*;
pub mod pip_optimiser;
pub use pip_optimiser::*;

pub struct PhyloOptimisationResult {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub tree: Tree,
    pub alignment: Alignment,
}

pub trait PhyloOptimiser<'a> {
    fn new(cost: &'a dyn PhyloCostFunction, info: &PhyloInfo) -> Self;
    fn run(self) -> Result<PhyloOptimisationResult>;
}

pub struct ModelOptimisationResult<M: EvoModel> {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub model: M,
}

pub trait ModelOptimiser<'a, LC: PhyloCostFunction, M: EvoModel> {
    fn new(cost: &'a LC, info: &PhyloInfo, frequencies: FrequencyOptimisation) -> Self;
    fn run(self) -> Result<ModelOptimisationResult<M>>;
}

#[cfg(test)]
mod blen_optimiser_tests;
#[cfg(test)]
mod pip_optimiser_tests;
#[cfg(test)]
mod subst_optimiser_tests;
