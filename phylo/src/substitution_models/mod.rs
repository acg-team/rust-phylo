use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::ops::Mul;

use anyhow::bail;
use nalgebra::{DMatrix, DVector};
use ordered_float::OrderedFloat;

use crate::alphabets::Alphabet;
use crate::evolutionary_models::EvoModel;
use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
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

pub struct SubstModelBuilder {
    model_id: String,
    freqs: Vec<f64>,
    params: Vec<f64>,
}

impl SubstModelBuilder {
    pub fn new(model_id: &str, freqs: &[f64], params: &[f64]) -> Self {
        SubstModelBuilder {
            model_id: model_id.to_string(),
            freqs: freqs.to_vec(),
            params: params.to_vec(),
        }
    }

    pub fn build(self) -> Result<Box<dyn EvoModel>> {
        match self.model_id.as_str() {
            "JC69" => Ok(Box::new(SubstModel::<JC69>::new(
                &self.freqs,
                &self.params,
            )?)),
            "K80" => Ok(Box::new(SubstModel::<K80>::new(&self.freqs, &self.params)?)),
            "HKY85" | "HKY" => Ok(Box::new(SubstModel::<HKY>::new(&self.freqs, &self.params)?)),
            "TN93" => Ok(Box::new(SubstModel::<TN93>::new(
                &self.freqs,
                &self.params,
            )?)),
            "GTR" => Ok(Box::new(SubstModel::<JC69>::new(
                &self.freqs,
                &self.params,
            )?)),
            "HIV" | "HIVB" => Ok(Box::new(SubstModel::<HIVB>::new(
                &self.freqs,
                &self.params,
            )?)),
            "WAG" => Ok(Box::new(SubstModel::<WAG>::new(&self.freqs, &self.params)?)),
            "BLOSUM" => Ok(Box::new(SubstModel::<BLOSUM>::new(
                &self.freqs,
                &self.params,
            )?)),
            _ => bail!("Unknown model requested."),
        }
    }
}

pub trait QMatrixFactory {
    fn new(frequencies: &[f64], params: &[f64]) -> Self;
}

pub trait QMatrix: Debug + Clone {
    fn set_param(&mut self, param: usize, value: f64);
    fn params(&self) -> &[f64];
    fn freqs(&self) -> &FreqVector;
    fn set_freqs(&mut self, freqs: FreqVector);
    fn q(&self) -> &SubstMatrix;
    fn rate(&self, i: u8, j: u8) -> f64;
    fn n(&self) -> usize;
    fn alphabet(&self) -> &Alphabet;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubstModel<Q>
where
    SubstModel<Q>: EvoModel,
    Q: QMatrix,
{
    pub(crate) qmatrix: Q,
}

pub trait ParsimonyModel {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)>;

    fn scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64);

    fn scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64);
}

impl<Q: QMatrix + QMatrixFactory + Display> Display for SubstModel<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.qmatrix)
    }
}

impl<Q: QMatrix + QMatrixFactory + Display> SubstModel<Q> {
    pub fn new(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(SubstModel {
            qmatrix: Q::new(frequencies, params),
        })
    }
}

impl<Q: QMatrix + QMatrixFactory + Display> EvoModel for SubstModel<Q> {
    fn p(&self, time: f64) -> SubstMatrix {
        (self.q().clone() * time).exp()
    }

    fn q(&self) -> &SubstMatrix {
        self.qmatrix.q()
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        self.qmatrix.rate(i, j)
    }

    fn params(&self) -> &[f64] {
        self.qmatrix.params()
    }

    fn set_param(&mut self, param: usize, value: f64) {
        self.qmatrix.set_param(param, value);
    }

    fn freqs(&self) -> &FreqVector {
        self.qmatrix.freqs()
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.qmatrix.set_freqs(pi);
    }

    fn n(&self) -> usize {
        self.qmatrix.n()
    }

    fn alphabet(&self) -> &Alphabet {
        self.qmatrix.alphabet()
    }
}

impl<Q: QMatrix> ParsimonyModel for SubstModel<Q>
where
    SubstModel<Q>: EvoModel,
{
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

pub struct SubstitutionCostBuilder {
    pub(crate) model: Box<dyn EvoModel>,
    info: PhyloInfo,
}

impl SubstitutionCostBuilder {
    pub fn new(model: Box<dyn EvoModel>, info: PhyloInfo) -> Self {
        SubstitutionCostBuilder { model, info }
    }

    pub fn build(self) -> Result<SubstitutionCost> {
        if self.info.msa.alphabet() != self.model.alphabet() {
            bail!("Alphabet mismatch between model and alignment.");
        }

        let tmp = RefCell::new(SubstModelInfo::new(&self.info, self.model.as_ref()).unwrap());
        Ok(SubstitutionCost {
            model: self.model,
            info: self.info,
            tmp,
        })
    }
}

#[derive(Debug)]
pub struct SubstitutionCost {
    pub(crate) model: Box<dyn EvoModel>,
    pub(crate) info: PhyloInfo,
    tmp: RefCell<SubstModelInfo>,
}

impl Clone for SubstitutionCost {
    fn clone(&self) -> Self {
        SubstitutionCost {
            model: self.model.clone(),
            info: self.info.clone(),
            tmp: RefCell::new(self.tmp.borrow().clone()),
        }
    }
}

impl TreeSearchCost for SubstitutionCost {
    fn cost(&self) -> f64 {
        self.logl(&self.info)
    }

    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]) {
        self.info.tree = tree;
        if dirty_nodes.is_empty() {
            self.tmp.borrow_mut().node_info_valid.fill(false);
            self.tmp.borrow_mut().node_models_valid.fill(false);
            return;
        }
        for node_idx in dirty_nodes {
            self.tmp.borrow_mut().node_info_valid[usize::from(node_idx)] = false;
            self.tmp.borrow_mut().node_models_valid[usize::from(node_idx)] = false;
        }
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

impl ModelSearchCost for SubstitutionCost {
    fn cost(&self) -> f64 {
        self.logl(&self.info)
    }

    fn set_param(&mut self, param: usize, value: f64) {
        self.model.set_param(param, value);
        self.tmp.borrow_mut().node_models_valid.fill(false);
    }

    fn params(&self) -> &[f64] {
        self.model.params()
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.model.set_freqs(freqs);
        self.tmp.borrow_mut().node_models_valid.fill(false);
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.model.freqs()
    }
}

impl Display for SubstitutionCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl SubstitutionCost {
    fn logl(&self, info: &PhyloInfo) -> f64 {
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

        let likelihood = self
            .model
            .freqs()
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
            tmp_values.node_models[idx] = self.model.p(node.blen);
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
        let node = tree.node(node_idx);
        let idx = usize::from(node_idx);

        if tree.dirty[idx] || !tmp_values.node_models_valid[idx] {
            tmp_values.node_models[idx] = self.model.p(node.blen);
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
pub struct SubstModelInfo {
    node_info: Vec<DMatrix<f64>>,
    node_info_valid: Vec<bool>,
    node_models: Vec<SubstMatrix>,
    node_models_valid: Vec<bool>,
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,
}

impl SubstModelInfo {
    pub fn new(info: &PhyloInfo, model: &dyn EvoModel) -> Result<Self> {
        let n = model.q().nrows();
        let node_count = info.tree.len();
        let msa_length = info.msa.len();

        let mut leaf_sequence_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.msa.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(n, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    site_info.copy_from(&leaf_encoding.column(c));
                } else {
                    site_info.copy_from(info.msa.alphabet().gap_encoding());
                }
            }
            leaf_sequence_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        Ok(SubstModelInfo {
            node_info: vec![DMatrix::<f64>::zeros(n, msa_length); node_count],
            node_info_valid: vec![false; node_count],
            node_models: vec![SubstMatrix::zeros(n, n); node_count],
            node_models_valid: vec![false; node_count],
            leaf_sequence_info,
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
pub(crate) mod tests;
