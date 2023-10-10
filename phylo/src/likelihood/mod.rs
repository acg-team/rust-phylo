use crate::{
    phylo_info::PhyloInfo,
    sequences::NUCLEOTIDES_STR,
    substitution_models::{DNASubstModel, SubstMatrix, SubstitutionModel},
    tree::NodeIdx,
    Result,
};
use nalgebra::{Const, DMatrix, DimMin};
use std::ops::Mul;

struct LikelihoodCostFunction<'a, const N: usize> {
    info: &'a PhyloInfo,
    model: SubstitutionModel<N>,
    temp_values: EvolutionaryModelInfo<N>,
}

struct EvolutionaryModelInfo<const N: usize> {
    internal_info: Vec<DMatrix<f64>>,
    internal_info_valid: Vec<bool>,
    internal_models: Vec<SubstMatrix<N>>,
    leaf_info: Vec<DMatrix<f64>>,
    leaf_info_valid: Vec<bool>,
    leaf_models: Vec<SubstMatrix<N>>,
    leaf_sequence_info: Vec<DMatrix<f64>>,
}

// implies that the sequences are aligned
impl<const N: usize> EvolutionaryModelInfo<N> {
    fn new(info: &PhyloInfo, model: &SubstitutionModel<N>) -> Self {
        let leaf_count = info.tree.leaves.len();
        let internal_count = info.tree.internals.len();
        let msa_length = info.sequences[0].seq().len();
        // set up basic char probabilities for the leaves
        let leaf_sequence_info = info
            .sequences
            .iter()
            .map(|rec| {
                DMatrix::from_fn(4, msa_length, |i, j| match rec.seq()[j] {
                    b'-' => model.pi[i],
                    _ => {
                        if NUCLEOTIDES_STR.find(rec.seq()[j] as char).unwrap() == i {
                            1.0
                        } else {
                            0.0
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        EvolutionaryModelInfo {
            internal_info: vec![DMatrix::<f64>::zeros(N, msa_length); internal_count],
            internal_info_valid: vec![false; internal_count],
            internal_models: vec![SubstMatrix::zeros(); internal_count],
            leaf_info: vec![DMatrix::<f64>::zeros(N, msa_length); leaf_count],
            leaf_info_valid: vec![false; leaf_count],
            leaf_models: vec![SubstMatrix::zeros(); leaf_count],
            leaf_sequence_info,
        }
    }
}

fn setup_dna_likelihood(
    info: &PhyloInfo,
    model_name: String,
    model_params: Vec<f64>,
    normalise: bool,
) -> Result<LikelihoodCostFunction<4>> {
    let mut model = DNASubstModel::new(&model_name, &model_params)?;
    if normalise {
        model.normalise();
    }
    let temp_values = EvolutionaryModelInfo::<4>::new(info, &model);
    Ok(LikelihoodCostFunction {
        info,
        model,
        temp_values,
    })
}

impl<'a, const N: usize> LikelihoodCostFunction<'a, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    pub fn compute_log_likelihood(&mut self) -> f64 {
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                NodeIdx::Internal(idx) => {
                    if !self.temp_values.internal_info_valid[*idx] {
                        let children = &self.info.tree.internals[*idx].children;
                        let (childx, childy) = (children[0], children[1]);
                        let childx_info = match childx {
                            NodeIdx::Internal(ch_idx) => &self.temp_values.internal_info[ch_idx],
                            NodeIdx::Leaf(ch_idx) => &self.temp_values.leaf_info[ch_idx],
                        };
                        let childy_info = match childy {
                            NodeIdx::Internal(ch_idx) => &self.temp_values.internal_info[ch_idx],
                            NodeIdx::Leaf(ch_idx) => &self.temp_values.leaf_info[ch_idx],
                        };
                        self.temp_values.internal_models[*idx] =
                            self.model.get_p(self.info.tree.internals[*idx].blen);
                        self.temp_values.internal_models[*idx].mul_to(
                            &(childx_info.component_mul(childy_info)),
                            &mut self.temp_values.internal_info[*idx],
                        );
                        self.temp_values.internal_info_valid[*idx] = true;
                    }
                }
                NodeIdx::Leaf(idx) => {
                    if !self.temp_values.leaf_info_valid[*idx] {
                        self.temp_values.leaf_models[*idx] =
                            self.model.get_p(self.info.tree.leaves[*idx].blen);
                        self.temp_values.leaf_models[*idx].mul_to(
                            &self.temp_values.leaf_sequence_info[*idx],
                            &mut self.temp_values.leaf_info[*idx],
                        );
                        self.temp_values.leaf_info_valid[*idx] = true;
                    }
                }
            };
        }

        let root_info = match self.info.tree.root {
            NodeIdx::Internal(idx) => &self.temp_values.internal_info[idx],
            NodeIdx::Leaf(idx) => &self.temp_values.leaf_info[idx],
        };
        let likelihood = self.model.pi.transpose().mul(root_info);
        assert_eq!(likelihood.ncols(), 1);
        assert_eq!(likelihood.nrows(), 1);
        likelihood[(0, 0)].ln()
    }
}

#[cfg(test)]
mod likelihood_tests;
