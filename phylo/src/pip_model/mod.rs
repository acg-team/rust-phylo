use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::Mul;
use std::vec;

use anyhow::bail;
use dyn_clone::DynClone;
use nalgebra::{DMatrix, DVector};

use crate::alignment::Mapping;
use crate::alphabets::{Alphabet, GAP};
use crate::evolutionary_models::EvoModel;
use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, QMatrix, QMatrixMaker, SubstMatrix};
use crate::tree::{
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

pub trait PIPModelTrait: Debug + Display + DynClone + EvoModel {
    fn lambda(&self) -> f64;
    fn mu(&self) -> f64;
}

dyn_clone::clone_trait_object!(PIPModelTrait);

// (2.0 * PI).ln() / 2.0;
pub static SHIFT: f64 = 0.9189385332046727;

fn log_factorial_shifted(n: usize) -> f64 {
    // An approximation using Stirling's formula, minus constant log(sqrt(2*PI)).
    // Has an absolute error of 3e-4 for n=2, 3e-6 for n=10, 3e-9 for n=100.
    let n = n as f64;
    n * n.ln() - n + n.ln() / 2.0 + 1.0 / (12.0 * n) + SHIFT
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModel<Q: QMatrix> {
    pub(crate) subst_q: Q,
    q: SubstMatrix,
    freqs: FreqVector,
    params: Vec<f64>,
}

fn pip_q(q: &mut SubstMatrix, subst_q: &SubstMatrix, mu: f64) {
    let n = subst_q.ncols();
    q.view_mut((0, 0), (n, n)).copy_from(subst_q);
    q.fill_column(n, mu);
    for i in 0..(n + 1) {
        q[(i, i)] -= mu;
    }
}

impl<Q: QMatrix> PIPModelTrait for PIPModel<Q> {
    fn lambda(&self) -> f64 {
        self.params[0]
    }

    fn mu(&self) -> f64 {
        self.params[1]
    }
}

impl<Q: QMatrix + QMatrixMaker> PIPModel<Q> {
    pub fn new(frequencies: &[f64], params: &[f64]) -> Result<Self> {
        if params.len() < 2 {
            bail!("Too few values provided for PIP, 2 values required, lambda and mu.");
        }
        let mu = params[1];

        let subst_q = Q::create(frequencies, &params[2..]);
        let n = subst_q.n();
        let freqs = subst_q.freqs().clone().insert_row(n, 0.0);
        let mut q = SubstMatrix::zeros(n + 1, n + 1);
        pip_q(&mut q, subst_q.q(), mu);
        Ok(PIPModel {
            subst_q,
            q,
            freqs,
            params: params.to_vec(),
        })
    }
}

impl<Q: QMatrix> EvoModel for PIPModel<Q> {
    fn p(&self, time: f64) -> SubstMatrix {
        (self.q().clone() * time).exp()
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        let n = self.subst_q.n();
        if i == GAP {
            self.q[(n, 0)]
        } else if j == GAP {
            self.q[(0, n)]
        } else {
            self.subst_q.rate(i, j)
        }
    }

    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }

    // This assumes correct dimensions to minimise runtime checks
    fn set_freqs(&mut self, pi: FreqVector) {
        debug_assert!(self.freqs.nrows() - 1 == pi.nrows() || self.freqs.nrows() == pi.nrows());
        self.freqs = pi.clone().insert_row(pi.nrows(), 0.0);
        self.subst_q.set_freqs(pi.clone());
        pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
    }

    fn params(&self) -> &[f64] {
        &self.params
    }

    fn set_param(&mut self, param: usize, value: f64) {
        match param {
            0 => {
                self.params[0] = value;
            }
            1 => {
                self.params[1] = value;
                pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
            }
            _ => {
                self.params[param] = value;
                self.subst_q.set_param(param - 2, value);
                pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
            }
        }
    }

    fn n(&self) -> usize {
        self.subst_q.n() + 1
    }

    fn alphabet(&self) -> &Alphabet {
        self.subst_q.alphabet()
    }
}

impl<Q: QMatrix + Display> Display for PIPModel<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PIP with [lambda = {:.5}, mu = {:.5}]\n and {}",
            self.params[0], self.params[1], self.subst_q
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,
    surv_ins_weights: Vec<f64>,
    anc: Vec<DMatrix<f64>>,
    ftilde: Vec<DMatrix<f64>>,
    f: Vec<DVector<f64>>,
    pnu: Vec<DVector<f64>>,
    c0_f1: Vec<f64>,
    c0_pnu: Vec<f64>,
    valid: Vec<bool>,
    models: Vec<SubstMatrix>,
    models_valid: Vec<bool>,
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,
}

impl<Q: QMatrix> PIPModelInfo<Q>
where
    PIPModel<Q>: PIPModelTrait,
{
    pub fn new(info: &PhyloInfo, model: &PIPModel<Q>) -> Result<Self> {
        let n = model.q().nrows();
        let node_count = info.tree.len();
        let msa_length = info.msa.len();
        let mut leaf_seq_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.msa.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(n, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = leaf_encoding.column(c).insert_row(n - 1, 0.0);
                    site_info.copy_from(&encoding);
                } else {
                    site_info.fill_row(n - 1, 1.0);
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_seq_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        Ok(PIPModelInfo {
            phantom: PhantomData,
            ftilde: vec![DMatrix::<f64>::zeros(n, msa_length); node_count],
            surv_ins_weights: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            pnu: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_f1: vec![0.0; node_count],
            c0_pnu: vec![0.0; node_count],
            anc: vec![DMatrix::<f64>::zeros(msa_length, 3); node_count],
            valid: vec![false; node_count],
            models: vec![SubstMatrix::zeros(n, n); node_count],
            models_valid: vec![false; node_count],
            leaf_sequence_info: leaf_seq_info,
        })
    }
}

pub struct PIPCostBuilder<Q: QMatrix>
where
    PIPModel<Q>: PIPModelTrait,
{
    pub(crate) model: PIPModel<Q>,
    info: PhyloInfo,
}

impl<Q: QMatrix> PIPCostBuilder<Q>
where
    PIPModel<Q>: PIPModelTrait,
{
    pub fn new(model: PIPModel<Q>, info: PhyloInfo) -> Self {
        PIPCostBuilder { model, info }
    }

    pub fn build(self) -> Result<PIPCost<Q>> {
        if self.info.msa.alphabet() != self.model.alphabet() {
            bail!("Alphabet mismatch between model and alignment.");
        }

        let tmp = RefCell::new(PIPModelInfo::new(&self.info, &self.model).unwrap());
        Ok(PIPCost {
            model: self.model,
            info: self.info,
            tmp,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PIPCost<Q: QMatrix> {
    pub(crate) model: PIPModel<Q>,
    pub(crate) info: PhyloInfo,
    tmp: RefCell<PIPModelInfo<Q>>,
}

impl<Q: QMatrix> TreeSearchCost for PIPCost<Q>
where
    PIPModel<Q>: PIPModelTrait,
{
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]) {
        self.info.tree = tree;
        if dirty_nodes.is_empty() {
            self.tmp.borrow_mut().valid.fill(false);
            self.tmp.borrow_mut().models_valid.fill(false);
            return;
        }
        for node_idx in dirty_nodes {
            self.tmp.borrow_mut().valid[usize::from(node_idx)] = false;
            self.tmp.borrow_mut().models_valid[usize::from(node_idx)] = false;
        }
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

impl<Q: QMatrix> ModelSearchCost for PIPCost<Q>
where
    PIPModel<Q>: PIPModelTrait,
{
    fn cost(&self) -> f64 {
        self.logl()
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.model.set_param(param, value);
        self.tmp.borrow_mut().models_valid.fill(false);
    }

    fn params(&self) -> &[f64] {
        self.model.params()
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.model.set_freqs(freqs);
        self.tmp.borrow_mut().models_valid.fill(false);
    }

    fn empirical_freqs(&self) -> FreqVector {
        self.info.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        self.model.freqs()
    }
}

impl<Q: QMatrix + Display> Display for PIPCost<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<Q: QMatrix> PIPCost<Q>
where
    PIPModel<Q>: PIPModelTrait,
{
    fn logl(&self) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Int(_) => {
                    if self.info.tree.root == *node_idx {
                        self.set_root(&self.info.tree, node_idx);
                    } else {
                        self.set_internal(&self.info.tree, node_idx);
                    }
                }
                Leaf(_) => {
                    self.set_leaf(&self.info.tree, node_idx, self.info.msa.leaf_map(node_idx));
                }
            };
        }

        let tmp = self.tmp.borrow();
        let root_idx = usize::from(&self.info.tree.root);
        let msa_length = self.info.msa.len();

        tmp.pnu[root_idx].map(|x| x.ln()).sum() + tmp.c0_pnu[root_idx]
            - log_factorial_shifted(msa_length)
    }

    fn set_root(&self, tree: &Tree, node_idx: &NodeIdx) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();

            tmp.surv_ins_weights[idx] = self.model.lambda() / self.model.mu();
            tmp.anc[idx].fill_column(0, 1.0);
            drop(tmp);

            self.set_ftilde(tree, node_idx);
            self.set_ancestors(tree, node_idx);
            self.set_pnu(tree, node_idx);
            self.set_c0(tree, node_idx);

            let mut tmp = self.tmp.borrow_mut();
            tmp.valid[idx] = true;
        }
    }

    fn set_internal(&self, tree: &Tree, node_idx: &NodeIdx) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        let node = tree.node(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();

            tmp.surv_ins_weights[idx] =
                Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node.blen);
            drop(tmp);

            self.set_ftilde(tree, node_idx);
            self.set_ancestors(tree, node_idx);
            self.set_pnu(tree, node_idx);
            self.set_c0(tree, node_idx);

            let mut tmp = self.tmp.borrow_mut();
            if let Some(parent_idx) = tree.parent(node_idx) {
                tmp.valid[usize::from(parent_idx)] = false;
            }
            tmp.valid[idx] = true;
        }
    }

    fn set_leaf(&self, tree: &Tree, node_idx: &NodeIdx, leaf_map: &Mapping) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        let node = tree.node(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();
            tmp.surv_ins_weights[idx] =
                Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node.blen);
            for (i, c) in leaf_map.iter().enumerate() {
                if c.is_some() {
                    tmp.anc[idx][(i, 0)] = 1.0;
                }
            }

            tmp.ftilde[idx] = tmp
                .leaf_sequence_info
                .get(tree.node_id(node_idx))
                .unwrap()
                .clone();

            tmp.f[idx] = tmp.ftilde[idx]
                .tr_mul(self.model.freqs())
                .component_mul(&tmp.anc[idx].column(0));

            tmp.pnu[idx] = tmp.f[idx].clone() * tmp.surv_ins_weights[idx];
            tmp.c0_f1[idx] = -1.0;
            tmp.c0_pnu[idx] = -tmp.surv_ins_weights[idx];
            if let Some(parent_idx) = tree.parent(node_idx) {
                tmp.valid[usize::from(parent_idx)] = false;
            }
            tmp.valid[idx] = true;
        }
    }

    fn set_pnu(&self, tree: &Tree, node_idx: &NodeIdx) {
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();
        let idx = usize::from(node_idx);

        let tmp = self.tmp.borrow();
        let x_pnu = tmp.pnu[children[0]].clone();
        let y_pnu = tmp.pnu[children[1]].clone();
        let ancestors = tmp.anc[idx].clone();

        drop(tmp);

        let mut tmp = self.tmp.borrow_mut();

        tmp.pnu[idx] =
            tmp.f[idx].clone().component_mul(&ancestors.column(0)) * tmp.surv_ins_weights[idx];
        tmp.pnu[idx] +=
            ancestors.column(1).component_mul(&x_pnu) + ancestors.column(2).component_mul(&y_pnu);
    }

    fn set_ftilde(&self, tree: &Tree, node_idx: &NodeIdx) {
        let children = tree.children(node_idx);
        let idx = usize::from(node_idx);
        let x_idx = usize::from(children[0]);
        let y_idx = usize::from(children[1]);
        let mut tmp = self.tmp.borrow_mut();
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = tmp.ftilde[idx].tr_mul(self.model.freqs());
    }

    fn set_model(&self, tree: &Tree, node_idx: &NodeIdx) {
        let idx = usize::from(node_idx);
        let node = tree.node(node_idx);
        if tree.dirty[idx] || !self.tmp.borrow().models_valid[idx] {
            let mut tmp = self.tmp.borrow_mut();
            tmp.models[idx] = self.model.p(node.blen);
            tmp.models_valid[idx] = true;
            tmp.valid[idx] = false;
        }
    }

    fn set_ancestors(&self, tree: &Tree, node_idx: &NodeIdx) {
        let idx = usize::from(node_idx);
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();
        let mut tmp = self.tmp.borrow_mut();
        let x_anc = tmp.anc[children[0]].clone();
        let y_anc = tmp.anc[children[1]].clone();
        tmp.anc[idx].set_column(1, &x_anc.column(0));
        tmp.anc[idx].set_column(2, &y_anc.column(0));
        tmp.anc[idx].set_column(0, &(x_anc.column(0) + y_anc.column(0)));
        for i in 0..tmp.anc[idx].nrows() {
            debug_assert!((0.0..=2.0).contains(&tmp.anc[idx][(i, 0)]));
            if tmp.anc[idx][(i, 0)] == 2.0 {
                tmp.anc[idx].fill_row(i, 0.0);
                tmp.anc[idx][(i, 0)] = 1.0;
            }
        }
    }

    fn set_c0(&self, tree: &Tree, node_idx: &NodeIdx) {
        let idx = usize::from(node_idx);
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();
        let x_idx = children[0];
        let y_idx = children[1];

        let x_blen = tree.nodes[x_idx].blen;
        let y_blen = tree.nodes[y_idx].blen;

        let mut tmp = self.tmp.borrow_mut();
        let mu = self.model.mu();
        tmp.c0_f1[idx] = (1.0 + (-mu * x_blen).exp() * tmp.c0_f1[x_idx])
            * (1.0 + (-mu * y_blen).exp() * tmp.c0_f1[y_idx])
            - 1.0;

        tmp.c0_pnu[idx] =
            tmp.surv_ins_weights[idx] * tmp.c0_f1[idx] + tmp.c0_pnu[x_idx] + tmp.c0_pnu[y_idx];
    }

    fn survival_insertion_weight(lambda: f64, mu: f64, b: f64) -> f64 {
        // A function equal to old
        // nu * insertion_probability(tree_length, b, mu) * survival_probablitily(mu, b)
        lambda / mu * (1.0 - (-b * mu).exp())
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
