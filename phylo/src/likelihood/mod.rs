use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{
    dna_models::DNASubstModel, protein_models::ProteinSubstModel, SubstitutionLikelihoodCost,
    SubstitutionModelInfo,
};
use crate::Result;

pub trait LikelihoodCostFunction<const N: usize> {
    fn compute_log_likelihood(&mut self) -> f64;
}

fn setup_dna_likelihood<'a>(
    info: &'a PhyloInfo,
    model_name: &str,
    model_params: &[f64],
    normalise: bool,
) -> Result<SubstitutionLikelihoodCost<'a, 4>> {
    let mut model = DNASubstModel::new(model_name, model_params, normalise)?;
    if normalise {
        model.normalise();
    }
    let temp_values = SubstitutionModelInfo::<4>::new(info, &model)?;
    Ok(SubstitutionLikelihoodCost {
        info,
        model,
        temp_values,
    })
}

fn setup_protein_likelihood<'a>(
    info: &'a PhyloInfo,
    model_name: &str,
    model_params: &[f64],
    normalise: bool,
) -> Result<SubstitutionLikelihoodCost<'a, 20>> {
    let mut model = ProteinSubstModel::new(model_name, model_params, normalise)?;
    if normalise {
        model.normalise();
    }
    let temp_values = SubstitutionModelInfo::<20>::new(info, &model)?;
    Ok(SubstitutionLikelihoodCost {
        info,
        model,
        temp_values,
    })
}

impl<'a, const N: usize> SubstitutionLikelihoodCost<'a, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn set_internal_values(&mut self, idx: &usize) {
        let node = &self.info.tree.internals[*idx];
        if !self.temp_values.internal_models_valid[*idx] {
            self.temp_values.internal_models[*idx] = self.model.get_p(node.blen);
            self.temp_values.internal_models_valid[*idx] = true;
        }
        let childx_info = self.child_info(&node.children[0]);
        let childy_info = self.child_info(&node.children[1]);
        self.temp_values.internal_models[*idx].mul_to(
            &(childx_info.component_mul(childy_info)),
            &mut self.temp_values.internal_info[*idx],
        );
        self.temp_values.internal_info_valid[*idx] = true;
    }

    fn set_child_values(&mut self, idx: &usize) {
        if !self.temp_values.leaf_models_valid[*idx] {
            self.temp_values.leaf_models[*idx] = self.model.get_p(self.info.tree.leaves[*idx].blen);
            self.temp_values.leaf_models_valid[*idx] = true;
        }
        self.temp_values.leaf_models[*idx].mul_to(
            &self.temp_values.leaf_sequence_info[*idx],
            &mut self.temp_values.leaf_info[*idx],
        );
        self.temp_values.leaf_info_valid[*idx] = true;
    }
}

impl<'a, const N: usize> SubstitutionLikelihoodCost<'a, N> {
    fn child_info(&self, child: &NodeIdx) -> &DMatrix<f64> {
        match child {
            NodeIdx::Internal(idx) => &self.temp_values.internal_info[*idx],
            NodeIdx::Leaf(idx) => &self.temp_values.leaf_info[*idx],
        }
    }
}

#[cfg(test)]
mod likelihood_tests;
