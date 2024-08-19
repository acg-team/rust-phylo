use crate::alignment::Alignment;
use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::tree::Tree;
use crate::Result;

pub mod branch_length_optimiser;
pub mod dna_model_optimiser;
pub mod pip_model_optimiser;

#[cfg(test)]
mod branch_length_optimiser_tests;
#[cfg(test)]
mod dna_optimisation_tests;
#[cfg(test)]
mod pip_dna_optimisation_tests;

pub struct PhyloOptimisationResult {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub tree: Tree,
    pub alignment: Alignment,
}

pub struct ModelOptimisationResult<M: EvoModel> {
    pub initial_logl: f64,
    pub final_logl: f64,
    pub iterations: usize,
    pub model: M,
}

pub trait ModelOptimiser<'a, LC: LikelihoodCostFunction, M: EvoModel> {
    fn new(cost: &'a LC, info: &PhyloInfo, frequencies: FrequencyOptimisation) -> Self;
    fn run(self) -> Result<ModelOptimisationResult<M>>;
}
