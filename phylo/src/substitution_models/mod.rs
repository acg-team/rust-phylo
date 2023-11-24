use std::collections::HashMap;
use std::ops::Mul;

use anyhow::bail;
use nalgebra::{Const, DMatrix, DVector, DimMin};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::tree::NodeIdx;
use crate::{f64_h, Result, Rounding};

pub mod dna_models;
pub mod protein_models;

pub type SubstMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[derive(Clone, Debug, PartialEq)]
pub struct SubstitutionModel<const N: usize> {
    index: [i32; 255],
    pub params: Vec<f64>,
    pub q: SubstMatrix,
    pub pi: FreqVector,
}

pub trait ParsimonyModel<const N: usize> {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)>;
    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64);
}

impl<const N: usize> SubstitutionModel<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn get_p(&self, time: f64) -> SubstMatrix {
        (self.q.clone() * time).exp()
    }

    pub(crate) fn get_rate(&self, i: u8, j: u8) -> f64 {
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
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        HashMap::<f64_h, (SubstMatrix, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.get_scoring_matrix_corrected(time, zero_diag, rounding),
            )
        }))
    }

    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix_corrected(time, false, rounding)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.pi
    }

    fn get_scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64) {
        let p = self.get_p(time);
        let mut scores = p.map(|x| -x.ln());
        if rounding.round {
            scores = scores.map(|x| {
                (x * 10.0_f64.powf(rounding.digits as f64)).round()
                    / 10.0_f64.powf(rounding.digits as f64)
            });
        }
        if zero_diag {
            scores.fill_diagonal(0.0);
        }
        let mean = scores.mean();
        (scores, mean)
    }
}

pub struct SubstitutionLikelihoodCost<'a, const N: usize> {
    pub info: &'a PhyloInfo,
    pub model: SubstitutionModel<N>,
    pub(crate) temp_values: SubstitutionModelInfo<N>,
}

pub(crate) struct SubstitutionModelInfo<const N: usize> {
    internal_info: Vec<DMatrix<f64>>,
    internal_info_valid: Vec<bool>,
    internal_models: Vec<SubstMatrix>,
    internal_models_valid: Vec<bool>,
    leaf_info: Vec<DMatrix<f64>>,
    leaf_info_valid: Vec<bool>,
    leaf_models: Vec<SubstMatrix>,
    leaf_models_valid: Vec<bool>,
    leaf_sequence_info: Vec<DMatrix<f64>>,
}

impl<const N: usize> EvolutionaryModelInfo<N> for SubstitutionModelInfo<N> {
    fn new(info: &PhyloInfo, model: &dyn EvolutionaryModel<N>) -> Result<Self> {
        if info.msa.is_none() {
            bail!("An MSA is required to set up the likelihood computation.");
        }
        let leaf_count = info.tree.leaves.len();
        let internal_count = info.tree.internals.len();
        let msa = info.msa.as_ref().unwrap();
        let msa_length = msa[0].seq().len();
        let leaf_sequence_info = msa
            .iter()
            .map(|rec| {
                DMatrix::from_columns(
                    rec.seq()
                        .iter()
                        .map(|&c| model.get_char_probability(c))
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
            })
            .collect::<Vec<_>>();
        Ok(SubstitutionModelInfo {
            internal_info: vec![DMatrix::<f64>::zeros(N, msa_length); internal_count],
            internal_info_valid: vec![false; internal_count],
            internal_models: vec![SubstMatrix::zeros(N, N); internal_count],
            internal_models_valid: vec![false; internal_count],
            leaf_info: vec![DMatrix::<f64>::zeros(N, msa_length); leaf_count],
            leaf_info_valid: vec![false; leaf_count],
            leaf_models: vec![SubstMatrix::zeros(N, N); leaf_count],
            leaf_models_valid: vec![false; leaf_count],
            leaf_sequence_info,
        })
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
                        self.set_leaf_values(idx);
                    }
                }
            };
        }
        let root_info = match self.info.tree.root {
            NodeIdx::Internal(idx) => &self.temp_values.internal_info[idx],
            NodeIdx::Leaf(idx) => &self.temp_values.leaf_info[idx],
        };
        let likelihood = self.model.pi.transpose().mul(root_info);
        debug_assert_eq!(
            likelihood.ncols(),
            self.info.msa.as_ref().unwrap()[0].seq().len()
        );
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
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

    fn set_leaf_values(&mut self, idx: &usize) {
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
pub(crate) mod substitution_models_tests;
