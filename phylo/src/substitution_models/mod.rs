use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Mul;

use nalgebra::{DMatrix, DVector};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvoModel, EvoModelParams};
use crate::likelihood::PhyloCostFunction;
use crate::tree::{
    Node,
    NodeIdx::{Internal, Leaf},
};
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

pub trait SubstModelParams {
    type ModelType;
    fn new(model_type: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized;
}

pub trait SubstitutionModel {
    type ModelType;
    type Params: SubstModelParams<ModelType = Self::ModelType> + EvoModelParams;
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
pub struct SubstModel<Params: SubstModelParams> {
    pub(crate) params: Params,
    pub(crate) q: SubstMatrix,
}

impl<Params: SubstModelParams> EvoModel for SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
    Params: SubstModelParams<ModelType = <SubstModel<Params> as SubstitutionModel>::ModelType>,
{
    type Params = Params;
    const N: usize = <SubstModel<Params> as SubstitutionModel>::N;

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

impl<Params: SubstModelParams> ParsimonyModel for SubstModel<Params>
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
pub struct SubstLikelihoodCost<'a, SubstModel: SubstitutionModel + 'a> {
    pub(crate) model: &'a SubstModel,
}

impl<'a, SubstModel: SubstitutionModel + EvoModel + 'a> PhyloCostFunction
    for SubstLikelihoodCost<'a, SubstModel>
{
    fn cost(&self, info: &PhyloInfo) -> f64 {
        self.logl(info).0
    }
}

impl<'a, SubstModel: SubstitutionModel + EvoModel> SubstLikelihoodCost<'a, SubstModel> {
    #[allow(dead_code)]
    pub(crate) fn new(model: &'a SubstModel) -> Self {
        SubstLikelihoodCost { model }
    }

    fn logl(&self, info: &PhyloInfo) -> (f64, SubstModelInfo<SubstModel>) {
        let mut tmp_info = SubstModelInfo::<SubstModel>::new(info, self.model).unwrap();
        let logl = self.logl_with_tmp(info, self.model, &mut tmp_info);
        (logl, tmp_info)
    }

    fn logl_with_tmp(
        &self,
        info: &PhyloInfo,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) -> f64 {
        debug_assert_eq!(info.tree.len(), tmp_values.node_info.len());
        for node_idx in info.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    self.set_internal(info.tree.node(node_idx), model, tmp_values);
                }
                Leaf(_) => {
                    self.set_leaf(info.tree.node(node_idx), model, tmp_values);
                }
            };
        }
        let root_info = &tmp_values.node_info[usize::from(&info.tree.root)];
        let likelihood = SubstitutionModel::freqs(model).transpose().mul(root_info);
        debug_assert_eq!(likelihood.ncols(), info.msa_length());
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
    }

    fn set_internal(
        &self,
        node: &Node,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) {
        let idx = usize::from(node.idx);
        if tmp_values.node_info_valid[idx] {
            return;
        }
        if !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = SubstitutionModel::p(model, node.blen);
            tmp_values.node_models_valid[idx] = true;
        }
        let childx_info = tmp_values.node_info[usize::from(&node.children[0])].clone();
        let childy_info = tmp_values.node_info[usize::from(&node.children[1])].clone();
        tmp_values.node_models[idx].mul_to(
            &(childx_info.component_mul(&childy_info)),
            &mut tmp_values.node_info[idx],
        );
        tmp_values.node_info_valid[idx] = true;
    }

    fn set_leaf(
        &self,
        node: &Node,
        model: &SubstModel,
        tmp_values: &mut SubstModelInfo<SubstModel>,
    ) {
        let idx = usize::from(node.idx);
        if tmp_values.node_info_valid[idx] {
            return;
        }
        if !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = SubstitutionModel::p(model, node.blen);
            tmp_values.node_models_valid[idx] = true;
        }
        tmp_values.node_models[idx].mul_to(
            tmp_values.leaf_sequence_info.get(&node.id).unwrap(),
            &mut tmp_values.node_info[idx],
        );
        tmp_values.node_info_valid[idx] = true;
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

impl<SubstModel: SubstitutionModel + EvoModel> SubstModelInfo<SubstModel> {
    pub fn new(info: &PhyloInfo, model: &SubstModel) -> Result<Self> {
        let node_count = info.tree.len();
        let msa_length = info.msa_length();

        let mut leaf_sequence_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps =
                DMatrix::<f64>::zeros(<SubstModel as SubstitutionModel>::N, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    site_info.copy_from(&leaf_encoding.column(c));
                    site_info.component_mul_assign(SubstitutionModel::freqs(model));
                } else {
                    site_info.copy_from(SubstitutionModel::freqs(model));
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_sequence_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        Ok(SubstModelInfo::<SubstModel> {
            phantom: PhantomData::<SubstModel>,
            node_info: vec![
                DMatrix::<f64>::zeros(<SubstModel as SubstitutionModel>::N, msa_length);
                node_count
            ],
            node_info_valid: vec![false; node_count],
            node_models: vec![
                SubstMatrix::zeros(
                    <SubstModel as SubstitutionModel>::N,
                    <SubstModel as SubstitutionModel>::N
                );
                node_count
            ],
            node_models_valid: vec![false; node_count],
            leaf_sequence_info,
        })
    }

    pub fn reset(&mut self) {
        self.node_info.iter_mut().for_each(|x| x.fill(0.0));
        self.node_info_valid.fill(false);
        self.node_models.iter_mut().for_each(|x| x.fill(0.0));
        self.node_models_valid.fill(false);
    }
}

#[cfg(test)]
pub(crate) mod tests;
