use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Mul;

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

impl<'a, SubstModel: SubstitutionModel + 'a> LikelihoodCostFunction
    for SubstitutionLikelihoodCost<'a, SubstModel>
{
    type Model = SubstModel;
    type Info = SubstModelInfo<SubstModel>;

    fn compute_logl(&self) -> f64 {
        self.compute_log_likelihood().0
    }
}

impl<'a, SubstModel: SubstitutionModel> SubstitutionLikelihoodCost<'a, SubstModel> {
    fn compute_log_likelihood_with_tmp(
        &self,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) -> f64 {
        debug_assert_eq!(self.info.tree.nodes.len(), tmp_values.node_info.len());
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                NodeIdx::Internal(idx) => {
                    if !tmp_values.node_info_valid[*idx] {
                        self.set_internal_values(idx, model, tmp_values);
                    }
                }
                NodeIdx::Leaf(idx) => {
                    if !tmp_values.node_info_valid[*idx] {
                        self.set_leaf_values(idx, model, tmp_values);
                    }
                }
            };
        }
        let root_info = &tmp_values.node_info[usize::from(&self.info.tree.root)];
        let likelihood = model.freqs().transpose().mul(root_info);
        debug_assert_eq!(likelihood.ncols(), self.info.msa_length());
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
    }

    fn set_internal_values(
        &self,
        idx: &usize,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) {
        let node = &self.info.tree.nodes[*idx];
        if !tmp_values.node_models_valid[*idx] {
            tmp_values.node_models[*idx] = SubstitutionModel::p(model, node.blen);
            tmp_values.node_models_valid[*idx] = true;
        }
        let childx_info = self.child_info(&node.children[0], tmp_values);
        let childy_info = self.child_info(&node.children[1], tmp_values);
        tmp_values.node_models[*idx].mul_to(
            &(childx_info.component_mul(&childy_info)),
            &mut tmp_values.node_info[*idx],
        );
        tmp_values.node_info_valid[*idx] = true;
    }

    fn set_leaf_values(
        &self,
        idx: &usize,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) {
        if !tmp_values.node_models_valid[*idx] {
            tmp_values.node_models[*idx] =
                SubstitutionModel::p(model, self.info.tree.nodes[*idx].blen);
            tmp_values.node_models_valid[*idx] = true;
        }
        tmp_values.node_models[*idx].mul_to(
            tmp_values
                .leaf_sequence_info
                .get(&self.info.tree.nodes[*idx].id)
                .unwrap(),
            &mut tmp_values.node_info[*idx],
        );
        tmp_values.node_info_valid[*idx] = true;
    }

    fn child_info(
        &self,
        child: &NodeIdx,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) -> DMatrix<f64> {
        tmp_values.node_info[usize::from(child)].clone()
    }
}

#[derive(Clone)]

pub struct SubstModelInfo<SubstModel: SubstitutionModel> {
    phantom: PhantomData<SubstModel>,
    node_info: Vec<DMatrix<f64>>,
    node_info_valid: Vec<bool>,
    node_models: Vec<SubstMatrix>,
    node_models_valid: Vec<bool>,
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,
}

impl<SubstModel: SubstitutionModel> EvoModelInfo for SubstModelInfo<SubstModel> {
    type Model = SubstModel;

    fn new(info: &PhyloInfo, model: &SubstModel) -> Result<Self>
    where
        Self: Sized,
    {
        Self::new(info, model)
    }

    fn reset(&mut self) {
        self.reset();
    }
}

impl<SubstModel: SubstitutionModel> SubstModelInfo<SubstModel> {
    fn new(info: &PhyloInfo, model: &SubstModel) -> Result<Self> {
        let node_count = info.tree.len();
        let msa_length = info.msa_length();

        let mut leaf_sequence_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(SubstModel::N, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    site_info.copy_from(&leaf_encoding.column(c));
                    site_info.component_mul_assign(model.freqs());
                } else {
                    site_info.copy_from(model.freqs());
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_sequence_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        Ok(SubstModelInfo::<SubstModel> {
            phantom: PhantomData::<SubstModel>,
            node_info: vec![DMatrix::<f64>::zeros(SubstModel::N, msa_length); node_count],
            node_info_valid: vec![false; node_count],
            node_models: vec![SubstMatrix::zeros(SubstModel::N, SubstModel::N); node_count],
            node_models_valid: vec![false; node_count],
            leaf_sequence_info,
        })
    }

    fn reset(&mut self) {
        self.node_info.iter_mut().for_each(|x| x.fill(0.0));
        self.node_info_valid.fill(false);
        self.node_models.iter_mut().for_each(|x| x.fill(0.0));
        self.node_models_valid.fill(false);
    }
}

#[cfg(test)]
pub(crate) mod substitution_models_tests;
