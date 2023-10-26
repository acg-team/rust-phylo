use crate::{
    phylo_info::PhyloInfo,
    substitution_models::{
        dna_models::DNASubstModel, SubstitutionLikelihoodCost, SubstitutionModel,
        SubstitutionModelInfo,
    },
    Result,
};

pub trait LikelihoodCostFunction<const N: usize> {
    fn compute_log_likelihood(&mut self) -> f64;
}

pub trait EvolutionaryModelInfo<const N: usize> {
    fn new(info: &PhyloInfo, model: &SubstitutionModel<N>) -> Self;
}

fn setup_dna_likelihood(
    info: &PhyloInfo,
    model_name: String,
    model_params: Vec<f64>,
    normalise: bool,
) -> Result<SubstitutionLikelihoodCost<4>> {
    let mut model = DNASubstModel::new(&model_name, &model_params)?;
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

#[cfg(test)]
mod likelihood_tests;
