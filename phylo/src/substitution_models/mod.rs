use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Mul;

use nalgebra::{DMatrix, DVector};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvoModel;
use crate::likelihood::PhyloCostFunction;
use crate::tree::NodeIdx;
use crate::tree::{
    NodeIdx::{Internal, Leaf},
    Tree,
};
use crate::{f64_h, phylo_info::PhyloInfo, Result, Rounding};

pub mod dna_models;
pub use dna_models::*;
pub mod protein_models;
pub use protein_models::*;

pub type SubstMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[macro_export]
macro_rules! frequencies {
    ($slice:expr) => {
        FreqVector::from_column_slice($slice)
    };
}

pub trait SubstitutionModel
where
    Self::ModelType: Display,
{
    type ModelType;
    const N: usize;

    fn new(model_type: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized;

    fn update(&mut self);

    fn normalise(&mut self);

    fn designation(&self) -> String;

    fn p(&self, time: f64) -> SubstMatrix {
        (self.q().clone() * time).exp()
    }

    fn q(&self) -> &SubstMatrix;

    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q()[(self.index()[i as usize], self.index()[j as usize])]
    }

    fn model_parameters(&self) -> Vec<f64>;

    fn set_param(&mut self, param: usize, value: f64);

    fn freqs(&self) -> &FreqVector;

    fn set_freqs(&mut self, freqs: FreqVector);

    fn index(&self) -> &[usize; 255];

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
pub struct SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    pub(crate) params: Params,
    pub(crate) q: SubstMatrix,
    tmp: RefCell<SubstModelInfo<SubstModel<Params>>>,
}

impl<Params: Display> Display for SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.params)
    }
}

impl<Params> EvoModel for SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    type ModelType = <SubstModel<Params> as SubstitutionModel>::ModelType;

    fn new(model_type: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        SubstitutionModel::new(model_type, params)
    }

    fn description(&self) -> String {
        SubstitutionModel::designation(self)
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

    fn model_parameters(&self) -> Vec<f64> {
        SubstitutionModel::model_parameters(self)
    }

    fn set_param(&mut self, param: usize, value: f64) {
        SubstitutionModel::set_param(self, param, value);
    }

    fn freqs(&self) -> &FreqVector {
        SubstitutionModel::freqs(self)
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        SubstitutionModel::set_freqs(self, pi);
    }

    fn index(&self) -> &[usize; 255] {
        SubstitutionModel::index(self)
    }
}

impl<Params> ParsimonyModel for SubstModel<Params>
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

impl<Params> PhyloCostFunction for SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    // TODO: add check that the model type matches the data
    fn cost(&self, info: &PhyloInfo, reset: bool) -> f64 {
        if reset {
            self.reset();
        }
        self.logl(info)
    }

    fn reset(&self) {
        self.tmp.borrow_mut().reset();
    }
}

impl<Params> SubstModel<Params>
where
    SubstModel<Params>: SubstitutionModel,
{
    fn normalise(&mut self) {
        let factor = -(SubstitutionModel::freqs(self).transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn logl(&self, info: &PhyloInfo) -> f64 {
        if self.tmp.borrow().empty {
            self.tmp
                .replace(SubstModelInfo::<SubstModel<Params>>::new(info, self).unwrap());
        }

        for node_idx in info.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    self.set_internal(&info.tree, node_idx);
                }
                Leaf(_) => {
                    self.set_leaf(&info.tree, node_idx);
                }
            };
        }
        let tmp_values = self.tmp.borrow();
        debug_assert_eq!(info.tree.len(), tmp_values.node_info.len());

        let likelihood = SubstitutionModel::freqs(self)
            .transpose()
            .mul(&tmp_values.node_info[usize::from(&info.tree.root)]);
        drop(tmp_values);

        debug_assert_eq!(likelihood.ncols(), info.msa.len());
        debug_assert_eq!(likelihood.nrows(), 1);
        likelihood.map(|x| x.ln()).sum()
    }

    fn set_internal(&self, tree: &Tree, node_idx: &NodeIdx) {
        let node = tree.node(node_idx);
        let childx_info = self.tmp.borrow().node_info[usize::from(&node.children[0])].clone();
        let childy_info = self.tmp.borrow().node_info[usize::from(&node.children[1])].clone();

        let idx = usize::from(node_idx);

        let mut tmp_values = self.tmp.borrow_mut();
        if tree.dirty[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = SubstitutionModel::p(self, node.blen);
            tmp_values.node_models_valid[idx] = true;
            tmp_values.node_info_valid[idx] = false;
        }
        if tmp_values.node_info_valid[idx] {
            return;
        }
        tmp_values.node_info[idx] =
            (&tmp_values.node_models[idx]).mul(childx_info.component_mul(&childy_info));
        tmp_values.node_info_valid[idx] = true;
        if let Some(parent_idx) = node.parent {
            tmp_values.node_info_valid[usize::from(parent_idx)] = false;
        }
        drop(tmp_values);
    }

    fn set_leaf(&self, tree: &Tree, node_idx: &NodeIdx) {
        let mut tmp_values = self.tmp.borrow_mut();
        let idx = usize::from(node_idx);

        if tree.dirty[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = SubstitutionModel::p(self, tree.blen(node_idx));
            tmp_values.node_models_valid[idx] = true;
            tmp_values.node_info_valid[idx] = false;
        }
        if tmp_values.node_info_valid[idx] {
            return;
        }

        // get leaf sequence encoding
        let leaf_seq = tmp_values
            .leaf_sequence_info
            .get(tree.node_id(node_idx))
            .unwrap();
        tmp_values.node_info[idx] = (&tmp_values.node_models[idx]).mul(leaf_seq);
        if let Some(parent_idx) = tree.parent(node_idx) {
            tmp_values.node_info_valid[usize::from(parent_idx)] = false;
        }
        tmp_values.node_info_valid[idx] = true;
        drop(tmp_values);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModelInfo<SM: SubstitutionModel> {
    empty: bool,
    phantom: PhantomData<SM>,
    node_info: Vec<DMatrix<f64>>,
    node_info_valid: Vec<bool>,
    node_models: Vec<SubstMatrix>,
    node_models_valid: Vec<bool>,
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,
}

impl<SM: SubstitutionModel> SubstModelInfo<SM> {
    pub fn empty() -> Self {
        SubstModelInfo::<SM> {
            empty: true,
            phantom: PhantomData::<SM>,
            node_info: Vec::new(),
            node_info_valid: Vec::new(),
            node_models: Vec::new(),
            node_models_valid: Vec::new(),
            leaf_sequence_info: HashMap::new(),
        }
    }

    pub fn new(info: &PhyloInfo, _model: &SM) -> Result<Self> {
        let node_count = info.tree.len();
        let msa_length = info.msa.len();

        let mut leaf_sequence_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.msa.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(SM::N, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    site_info.copy_from(&leaf_encoding.column(c));
                } else {
                    site_info.copy_from(info.msa.alphabet().gap_encoding());
                }
            }
            leaf_sequence_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        Ok(SubstModelInfo::<SM> {
            empty: false,
            phantom: PhantomData::<SM>,
            node_info: vec![DMatrix::<f64>::zeros(SM::N, msa_length); node_count],
            node_info_valid: vec![false; node_count],
            node_models: vec![SubstMatrix::zeros(SM::N, SM::N); node_count],
            node_models_valid: vec![false; node_count],
            leaf_sequence_info,
        })
    }

    pub fn reset(&mut self) {
        self.empty = true;
        self.node_info.iter_mut().for_each(|x| x.fill(0.0));
        self.node_info_valid.fill(false);
        self.node_models.iter_mut().for_each(|x| x.fill(0.0));
        self.node_models_valid.fill(false);
    }
}

#[cfg(test)]
pub(crate) mod tests;
