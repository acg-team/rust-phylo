use crate::evolutionary_models::EvolutionaryModel;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{
    dna_models::DNASubstModel, protein_models::ProteinSubstModel, SubstitutionLikelihoodCost,
    SubstitutionModelInfo,
};
use crate::Result;

pub trait LikelihoodCostFunction<const N: usize> {
    fn compute_log_likelihood(&mut self) -> f64;
}

pub trait EvolutionaryModelInfo<const N: usize> {
    fn new(info: &PhyloInfo, model: &dyn EvolutionaryModel<N>) -> Self;
}

fn setup_dna_likelihood<'a>(
    info: &'a PhyloInfo,
    model_name: String,
    model_params: &[f64],
    normalise: bool,
) -> Result<SubstitutionLikelihoodCost<'a, 4>> {
    let mut model = DNASubstModel::new(&model_name, model_params)?;
    if normalise {
        model.normalise();
    }
    let temp_values = SubstitutionModelInfo::<4>::new(info, &model);
    Ok(SubstitutionLikelihoodCost {
        info,
        model,
        temp_values,
    })
}

fn setup_protein_likelihood(
    info: &PhyloInfo,
    model_name: String,
    normalise: bool,
) -> Result<SubstitutionLikelihoodCost<20>> {
    let mut model = ProteinSubstModel::new(&model_name, &[])?;
    if normalise {
        model.normalise();
    }
    let temp_values = SubstitutionModelInfo::<20>::new(info, &model);
    Ok(SubstitutionLikelihoodCost {
        info,
        model,
        temp_values,
    })
}

#[cfg(test)]
mod likelihood_tests;
