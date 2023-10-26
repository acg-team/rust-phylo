use crate::{
    f64_h,
    likelihood::{EvolutionaryModelInfo, LikelihoodCostFunction},
    phylo_info::PhyloInfo,
    sequences::NUCLEOTIDES_STR,
    tree::NodeIdx,
};
use nalgebra::{Const, DMatrix, DimMin, SMatrix, SVector};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::ops::Mul;

pub mod dna_models;
pub mod protein_models;

pub type SubstMatrix<const N: usize> = SMatrix<f64, N, N>;
pub type FreqVector<const N: usize> = SVector<f64, N>;

#[derive(Clone, Debug, PartialEq)]
pub struct SubstitutionModel<const N: usize> {
    index: [i32; 255],
    pub q: SubstMatrix<N>,
    pub pi: FreqVector<N>,
}

impl<const N: usize> SubstitutionModel<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    pub fn get_p(&self, time: f64) -> SubstMatrix<N> {
        (self.q * time).exp()
    }

    pub fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    pub fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix<N>, f64)> {
        HashMap::<f64_h, (SubstMatrix<N>, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.get_scoring_matrix_corrected(time, zero_diag, rounded),
            )
        }))
    }

    pub fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    pub fn get_scoring_matrix(&self, time: f64, rounded: bool) -> (SubstMatrix<N>, f64) {
        self.get_scoring_matrix_corrected(time, false, rounded)
    }

    fn get_scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounded: bool,
    ) -> (SubstMatrix<N>, f64) {
        let p = self.get_p(time);
        let mapping = if rounded {
            |x: f64| (-x.ln().round())
        } else {
            |x: f64| -x.ln()
        };
        let mut scores = p.map(mapping);
        if zero_diag {
            scores.fill_diagonal(0.0);
        }
        (scores, scores.mean())
    }
}

pub(crate) struct SubstitutionLikelihoodCost<'a, const N: usize> {
    pub(crate) info: &'a PhyloInfo,
    pub(crate) model: SubstitutionModel<N>,
    pub(crate) temp_values: SubstitutionModelInfo<N>,
}

pub(crate) struct SubstitutionModelInfo<const N: usize> {
    internal_info: Vec<DMatrix<f64>>,
    internal_info_valid: Vec<bool>,
    internal_models: Vec<SubstMatrix<N>>,
    internal_models_valid: Vec<bool>,
    leaf_info: Vec<DMatrix<f64>>,
    leaf_info_valid: Vec<bool>,
    leaf_models: Vec<SubstMatrix<N>>,
    leaf_models_valid: Vec<bool>,
    leaf_sequence_info: Vec<DMatrix<f64>>,
}

// TODO: Convert SubstitutionModel to a trait EvolutionaryModel, and implement it for DNASubstModel, ProteinSubstModel, and PIP
// implies that the sequences are aligned
impl<const N: usize> EvolutionaryModelInfo<N> for SubstitutionModelInfo<N> {
    fn new(info: &PhyloInfo, model: &SubstitutionModel<N>) -> Self {
        let leaf_count = info.tree.leaves.len();
        let internal_count = info.tree.internals.len();
        let msa_length = info.sequences[0].seq().len();
        // set up basic char probabilities for the leaves
        let leaf_sequence_info = info
            .sequences
            .iter()
            .map(|rec| {
                // This is incorrect (ignoress ambig chars) and only works for DNA
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
        SubstitutionModelInfo {
            internal_info: vec![DMatrix::<f64>::zeros(N, msa_length); internal_count],
            internal_info_valid: vec![false; internal_count],
            internal_models: vec![SubstMatrix::zeros(); internal_count],
            internal_models_valid: vec![false; internal_count],
            leaf_info: vec![DMatrix::<f64>::zeros(N, msa_length); leaf_count],
            leaf_info_valid: vec![false; leaf_count],
            leaf_models: vec![SubstMatrix::zeros(); leaf_count],
            leaf_models_valid: vec![false; leaf_count],
            leaf_sequence_info,
        }
    }
}

impl<const N: usize> LikelihoodCostFunction<N> for SubstitutionLikelihoodCost<'_, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn compute_log_likelihood(&mut self) -> f64 {
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                NodeIdx::Internal(idx) => {
                    if !self.temp_values.internal_info_valid[*idx] {
                        self.set_internal_values(idx);
                    }
                }
                NodeIdx::Leaf(idx) => {
                    if !self.temp_values.leaf_info_valid[*idx] {
                        self.set_child_values(idx);
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

    fn child_info(&self, child: &NodeIdx) -> &DMatrix<f64> {
        match child {
            NodeIdx::Internal(idx) => &self.temp_values.internal_info[*idx],
            NodeIdx::Leaf(idx) => &self.temp_values.leaf_info[*idx],
        }
    }
}

#[cfg(test)]
mod substitution_models_tests;
