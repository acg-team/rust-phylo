use crate::alignment::Alignment;
use crate::tree::Tree;

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
