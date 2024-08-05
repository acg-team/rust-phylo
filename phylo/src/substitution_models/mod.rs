use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Mul;

use anyhow::bail;
use nalgebra::{DMatrix, DVector};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvoModelInfo, EvoModelParams, EvolutionaryModel};
use crate::likelihood::LikelihoodCostFunction;
use crate::tree::NodeIdx;
use crate::{f64_h, phylo_info::PhyloInfo, Result, Rounding};

pub mod dna_models;
pub mod protein_models;

pub type SubstMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[macro_export]
macro_rules! frequencies {
    ($slice:expr) => {
        FreqVector::from_column_slice($slice)
    };
}

pub trait SubstitutionModel {
    type ModelType;
    type Params: EvoModelParams<ModelType = Self::ModelType>;
    const N: usize;
    const ALPHABET: &'static [u8];

    fn char_sets() -> &'static [FreqVector];
    fn create(params: &Self::Params) -> Self;
    fn new(model_type: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized;
    fn index(&self) -> &[usize; 255];
    fn freqs(&self) -> &FreqVector;
    fn q(&self) -> &SubstMatrix;
    fn normalise(&mut self);

    fn p(&self, time: f64) -> SubstMatrix {
        (self.q().clone() * time).exp()
    }
    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q()[(self.index()[i as usize], self.index()[j as usize])]
    }
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        HashMap::<f64_h, (SubstMatrix, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.scoring_matrix_corrected(time, zero_diag, rounding),
            )
        }))
    }
    fn scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.scoring_matrix_corrected(time, false, rounding)
    }
    fn scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64) {
        let mut scores = self.p(time);
        scores.iter_mut().for_each(|x| *x = -(*x).ln());
        if rounding.round {
            scores.iter_mut().for_each(|x| {
                *x = (*x * 10.0_f64.powf(rounding.digits as f64)).round()
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

pub trait ParsimonyModel {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)>;
    fn scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64);
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModel<Params: EvoModelParams> {
    pub(crate) params: Params,
    pub(crate) q: SubstMatrix,
}

impl<Params: EvoModelParams> EvolutionaryModel for SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    type ModelType = <SubstModel<Params> as SubstitutionModel>::ModelType;
    type Params = Params;

    fn new(model: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        SubstitutionModel::new(model, params)
    }

    fn p(&self, time: f64) -> SubstMatrix {
        SubstitutionModel::p(self, time)
    }

    fn q(&self) -> &SubstMatrix {
        SubstitutionModel::q(self)
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        SubstitutionModel::rate(self, i, j)
    }

    fn freqs(&self) -> &FreqVector {
        SubstitutionModel::freqs(self)
    }

    fn char_probability(&self, char_encoding: &FreqVector) -> FreqVector {
        let mut probs = SubstitutionModel::freqs(self)
            .clone()
            .component_mul(char_encoding);
        probs.scale_mut(1.0 / probs.sum());
        probs
    }

    fn index(&self) -> &[usize; 255] {
        SubstitutionModel::index(self)
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }
}

impl<Params: EvoModelParams> ParsimonyModel for SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        SubstitutionModel::generate_scorings(self, times, zero_diag, rounding)
    }

    fn scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        SubstitutionModel::scoring_matrix(self, time, rounding)
    }
}

#[derive(Clone)]
pub struct SubstitutionLikelihoodCost<'a, SubstModel: SubstitutionModel + 'a> {
    pub(crate) info: &'a PhyloInfo,
    pub(crate) model: &'a SubstModel,
}

impl<'a, SubstModel: SubstitutionModel + 'a> SubstitutionLikelihoodCost<'a, SubstModel> {
    pub fn new(info: &'a PhyloInfo, model: &'a SubstModel) -> Self {
        SubstitutionLikelihoodCost { info, model }
    }
    pub fn compute_log_likelihood(&self) -> (f64, SubstModelInfo<SubstModel>) {
        let mut tmp_info = SubstModelInfo::<SubstModel>::new(self.info, self.model).unwrap();
        let logl = self.compute_log_likelihood_with_tmp(self.model, &mut tmp_info);
        (logl, tmp_info)
    }
}

impl<'a, SubstModel: SubstitutionModel + 'a> LikelihoodCostFunction<'a>
    for SubstitutionLikelihoodCost<'a, SubstModel>
{
    type Model = SubstModel;
    type Info = SubstModelInfo<SubstModel>;

    fn compute_logl(&self) -> f64 {
        self.compute_log_likelihood().0
    }

    fn empirical_frequencies(&self) -> FreqVector {
        let all_counts = self.info.counts();
        let mut total = all_counts.values().sum::<f64>();
        let index = SubstitutionModel::index(self.model);
        let mut freqs = FreqVector::from_column_slice(&vec![0.0; Self::Model::N]);
        for (&char, &count) in all_counts.iter() {
            freqs += &Self::Model::char_sets()[char as usize].scale(count);
        }
        for &char in Self::Model::ALPHABET {
            if freqs[index[char as usize]] == 0.0 {
                freqs[index[char as usize]] += 1.0;
                total += 1.0;
            }
        }
        let scaler = 1.0 / total;
        freqs.scale_mut(scaler);
        freqs
    }
}

impl<'a, SubstModel: SubstitutionModel> SubstitutionLikelihoodCost<'a, SubstModel> {
    fn compute_log_likelihood_with_tmp(
        &self,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) -> f64 {
        debug_assert_eq!(
            self.info.tree.internals.len(),
            tmp_values.internal_info.len()
        );
        debug_assert_eq!(self.info.tree.leaves.len(), tmp_values.leaf_info.len());
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                NodeIdx::Internal(idx) => {
                    if !tmp_values.internal_info_valid[*idx] {
                        self.set_internal_values(idx, model, tmp_values);
                    }
                }
                NodeIdx::Leaf(idx) => {
                    if !tmp_values.leaf_info_valid[*idx] {
                        self.set_leaf_values(idx, model, tmp_values);
                    }
                }
            };
        }
        let root_info = match self.info.tree.root {
            NodeIdx::Internal(idx) => &tmp_values.internal_info[idx],
            NodeIdx::Leaf(idx) => &tmp_values.leaf_info[idx],
        };
        let likelihood = model.freqs().transpose().mul(root_info);
        debug_assert_eq!(
            likelihood.ncols(),
            self.info.msa.as_ref().unwrap()[0].seq().len()
        );
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
    }

    fn set_internal_values(
        &self,
        idx: &usize,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) {
        let node = &self.info.tree.internals[*idx];
        if !tmp_values.internal_models_valid[*idx] {
            tmp_values.internal_models[*idx] = SubstitutionModel::p(model, node.blen);
            tmp_values.internal_models_valid[*idx] = true;
        }
        let childx_info = self.child_info(&node.children[0], tmp_values);
        let childy_info = self.child_info(&node.children[1], tmp_values);
        tmp_values.internal_models[*idx].mul_to(
            &(childx_info.component_mul(&childy_info)),
            &mut tmp_values.internal_info[*idx],
        );
        tmp_values.internal_info_valid[*idx] = true;
    }

    fn set_leaf_values(
        &self,
        idx: &usize,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) {
        if !tmp_values.leaf_models_valid[*idx] {
            tmp_values.leaf_models[*idx] =
                SubstitutionModel::p(model, self.info.tree.leaves[*idx].blen);
            tmp_values.leaf_models_valid[*idx] = true;
        }
        tmp_values.leaf_models[*idx].mul_to(
            &tmp_values.leaf_sequence_info[*idx],
            &mut tmp_values.leaf_info[*idx],
        );
        tmp_values.leaf_info_valid[*idx] = true;
    }

    fn child_info(
        &self,
        child: &NodeIdx,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) -> DMatrix<f64> {
        match child {
            NodeIdx::Internal(idx) => tmp_values.internal_info[*idx].clone(),
            NodeIdx::Leaf(idx) => tmp_values.leaf_info[*idx].clone(),
        }
    }
}

#[derive(Clone)]

pub struct SubstModelInfo<SubstModel: SubstitutionModel> {
    phantom: PhantomData<SubstModel>,
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

impl<SubstModel: SubstitutionModel> EvoModelInfo for SubstModelInfo<SubstModel> {
    type Model = SubstModel;

    fn new(info: &PhyloInfo, model: &SubstModel) -> Result<Self>
    where
        Self: Sized,
    {
        if info.msa.is_none() {
            bail!("An MSA is required to set up the likelihood computation.");
        }
        let leaf_count = info.tree.leaves.len();
        let internal_count = info.tree.internals.len();
        let msa = info.msa.as_ref().unwrap();
        let msa_length = msa[0].seq().len();

        let mut leaf_sequence_info = info.leaf_encoding.clone();
        for leaf_seq in leaf_sequence_info.iter_mut() {
            for mut site_info in leaf_seq.column_iter_mut() {
                site_info.component_mul_assign(model.freqs());
                site_info.scale_mut((1.0) / site_info.sum());
            }
        }
        Ok(SubstModelInfo::<SubstModel> {
            phantom: PhantomData::<SubstModel>,
            internal_info: vec![DMatrix::<f64>::zeros(SubstModel::N, msa_length); internal_count],
            internal_info_valid: vec![false; internal_count],
            internal_models: vec![SubstMatrix::zeros(SubstModel::N, SubstModel::N); internal_count],
            internal_models_valid: vec![false; internal_count],
            leaf_info: vec![DMatrix::<f64>::zeros(SubstModel::N, msa_length); leaf_count],
            leaf_info_valid: vec![false; leaf_count],
            leaf_models: vec![SubstMatrix::zeros(SubstModel::N, SubstModel::N); leaf_count],
            leaf_models_valid: vec![false; leaf_count],
            leaf_sequence_info,
        })
    }

    fn reset(&mut self) {
        self.internal_info.iter_mut().for_each(|x| x.fill(0.0));
        self.internal_info_valid.fill(false);
        self.internal_models.iter_mut().for_each(|x| x.fill(0.0));
        self.internal_models_valid.fill(false);
        self.leaf_info.iter_mut().for_each(|x| x.fill(0.0));
        self.leaf_info_valid.fill(false);
        self.leaf_models.iter_mut().for_each(|x| x.fill(0.0));
        self.leaf_models_valid.fill(false);
    }
}

#[cfg(test)]
pub(crate) mod substitution_models_tests;
