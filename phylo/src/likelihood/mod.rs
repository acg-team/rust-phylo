use crate::{
    phylo_info::PhyloInfo,
    substitution_models::{
        dna_models::DNASubstModel, EvolutionaryModel, SubstitutionLikelihoodCost,
        SubstitutionModelInfo,
    },
    Result,
};

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

#[cfg(test)]
mod likelihood_tests;
