use anyhow::bail;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Mul;
use std::vec;

use crate::alignment::Mapping;
use crate::evolutionary_models::EvoModel;
use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{FreqVector, QMatrix, SubstMatrix};
use crate::tree::{
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModel<Q: QMatrix> {
    pub(crate) subst_q: Q,
    q: SubstMatrix,
    freqs: FreqVector,
    index: [usize; 255],
    params: Vec<f64>,
}

impl<Q: QMatrix + Clone> PIPModel<Q> {
    fn lambda(&self) -> f64 {
        self.params[0]
    }
    fn mu(&self) -> f64 {
        self.params[1]
    }
}

fn pip_q(q: &mut SubstMatrix, subst_q: &SubstMatrix, mu: f64) {
    let n = subst_q.ncols();
    q.view_mut((0, 0), (n, n)).copy_from(subst_q);
    q.fill_column(n, mu);
    for i in 0..(n + 1) {
        q[(i, i)] -= mu;
    }
}

impl<Q: QMatrix + Clone> EvoModel for PIPModel<Q> {
    fn new(frequencies: &[f64], params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        if params.len() < 2 {
            bail!("Too few values provided for PIP, 2 values required, lambda and mu.");
        }
        let mu = params[1];

        let subst_q = Q::new(frequencies, &params[2..]);
        let mut index = *subst_q.index();
        index[b'-' as usize] = subst_q.n();
        let freqs = subst_q.freqs().clone().insert_row(subst_q.n(), 0.0);
        let mut q = SubstMatrix::zeros(subst_q.n() + 1, subst_q.n() + 1);
        pip_q(&mut q, subst_q.q(), mu);
        Ok(PIPModel {
            subst_q,
            q,
            freqs,
            index,
            params: params.to_vec(),
        })
    }

    fn p(&self, time: f64) -> SubstMatrix {
        (self.q().clone() * time).exp()
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(self.index[i as usize], self.index[j as usize])]
    }

    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.freqs = pi.clone().insert_row(self.n() - 1, 0.0);
        self.subst_q.set_freqs(pi.clone());
        pip_q(&mut self.q, self.subst_q.q(), self.params[1]);
    }

    fn index(&self) -> &[usize; 255] {
        &self.index
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
    ins_probs: Vec<f64>,
    surv_probs: Vec<f64>,
    anc: Vec<DMatrix<f64>>,
    ftilde: Vec<DMatrix<f64>>,
    f: Vec<DVector<f64>>,
    p: Vec<DVector<f64>>,
    c0_ftilde: Vec<DMatrix<f64>>,
    c0_f: Vec<f64>,
    c0_p: Vec<f64>,
    valid: Vec<bool>,
    models: Vec<SubstMatrix>,
    models_valid: Vec<bool>,
    leaf_sequence_info: HashMap<String, DMatrix<f64>>,
}

impl<Q: QMatrix + Clone> PIPModelInfo<Q>
where
    PIPModel<Q>: EvoModel,
{
    pub fn new(info: &PhyloInfo, model: &PIPModel<Q>) -> Self {
        let n = model.n();
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

        PIPModelInfo::<Q> {
            phantom: PhantomData,
            ftilde: vec![DMatrix::<f64>::zeros(model.n(), msa_length); node_count],
            ins_probs: vec![0.0; node_count],
            surv_probs: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            p: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_ftilde: vec![DMatrix::<f64>::zeros(n, 1); node_count],
            c0_f: vec![0.0; node_count],
            c0_p: vec![0.0; node_count],
            anc: vec![DMatrix::<f64>::zeros(msa_length, 3); node_count],
            valid: vec![false; node_count],
            models: vec![SubstMatrix::zeros(n, n); node_count],
            models_valid: vec![false; node_count],
            leaf_sequence_info: leaf_seq_info,
        }
    }
}

pub struct PIPCostBuilder<Q: QMatrix + Display> {
    model: PIPModel<Q>,
    info: PhyloInfo,
}

impl<Q: QMatrix + Display + Clone> PIPCostBuilder<Q> {
    pub fn new(model: PIPModel<Q>, info: PhyloInfo) -> Self {
        PIPCostBuilder { model, info }
    }

    pub fn build(self) -> Result<PIPCost<Q>> {
        if self.info.msa.alphabet() != self.model.subst_q.alphabet() {
            bail!("Alphabet mismatch between model and alignment.");
        }

        let tmp = RefCell::new(PIPModelInfo::<Q>::new(&self.info, &self.model));
        Ok(PIPCost {
            model: self.model,
            info: self.info,
            tmp,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PIPCost<Q: QMatrix + Display + 'static> {
    pub(crate) model: PIPModel<Q>,
    pub(crate) info: PhyloInfo,
    tmp: RefCell<PIPModelInfo<Q>>,
}

impl<Q: QMatrix + Display + Clone + 'static> TreeSearchCost for PIPCost<Q>
where
    PIPModel<Q>: EvoModel,
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

impl<Q: QMatrix + Display + Clone + 'static> ModelSearchCost for PIPCost<Q>
where
    PIPModel<Q>: EvoModel,
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

impl<Q: QMatrix + Display + Clone + 'static> Display for PIPCost<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<Q: QMatrix + Display + Clone> PIPCost<Q>
where
    PIPModel<Q>: EvoModel,
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
        let nu = self.model.lambda() * (self.info.tree.height + 1.0 / self.model.mu());
        let ln_phi = nu.ln() * msa_length as f64 + (tmp.c0_p[root_idx] - 1.0) * nu
            - (log_factorial(msa_length));
        tmp.p[root_idx].map(|x| x.ln()).sum() + ln_phi
    }

    fn set_root(&self, tree: &Tree, node_idx: &NodeIdx) {
        self.set_model(tree, node_idx);
        let idx = usize::from(node_idx);
        if !self.tmp.borrow().valid[idx] {
            let mut tmp = self.tmp.borrow_mut();
            let mu = self.model.mu();
            tmp.surv_probs[idx] = 1.0;
            tmp.ins_probs[idx] = Self::insertion_prob(tree.height, 1.0 / mu, mu);
            tmp.anc[idx].fill_column(0, 1.0);
            drop(tmp);

            self.set_ftilde(tree, node_idx);
            self.set_ancestors(tree, node_idx);
            self.set_p(tree, node_idx);
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
            let mu = self.model.mu();
            tmp.surv_probs[idx] = Self::survival_prob(mu, node.blen);
            tmp.ins_probs[idx] = Self::insertion_prob(tree.height, node.blen, mu);
            drop(tmp);

            self.set_ftilde(tree, node_idx);
            self.set_ancestors(tree, node_idx);
            self.set_p(tree, node_idx);
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
            let mu = self.model.mu();
            tmp.surv_probs[idx] = Self::survival_prob(mu, node.blen);
            tmp.ins_probs[idx] = Self::insertion_prob(tree.height, node.blen, mu);
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
            tmp.p[idx] = tmp.f[idx].clone() * tmp.surv_probs[idx] * tmp.ins_probs[idx];
            tmp.c0_ftilde[idx][self.model.n() - 1] = 1.0;
            tmp.c0_f[idx] = 0.0;
            tmp.c0_p[idx] = (1.0 - tmp.surv_probs[idx]) * tmp.ins_probs[idx];

            if let Some(parent_idx) = tree.parent(node_idx) {
                tmp.valid[usize::from(parent_idx)] = false;
            }
            tmp.valid[idx] = true;
        }
    }

    fn set_p(&self, tree: &Tree, node_idx: &NodeIdx) {
        let children: Vec<usize> = tree.children(node_idx).iter().map(usize::from).collect();

        let idx = usize::from(node_idx);

        let tmp = self.tmp.borrow();
        let anc_x = tmp.anc[idx]
            .column(1)
            .component_mul(&tmp.p[children[0]].clone());
        let anc_y = tmp.anc[idx]
            .column(2)
            .component_mul(&tmp.p[children[1]].clone());
        drop(tmp);

        let mut tmp = self.tmp.borrow_mut();
        tmp.p[idx] = tmp.f[idx].clone().component_mul(&tmp.anc[idx].column(0))
            * tmp.surv_probs[idx]
            * tmp.ins_probs[idx];
        tmp.p[idx] += anc_x + anc_y;
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

        let mut tmp = self.tmp.borrow_mut();
        tmp.c0_ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.c0_ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.c0_ftilde[y_idx]));
        tmp.c0_f[idx] = tmp.c0_ftilde[idx].tr_mul(self.model.freqs())[0];
        tmp.c0_p[idx] = (1.0 + tmp.surv_probs[idx] * (tmp.c0_f[idx] - 1.0)) * tmp.ins_probs[idx]
            + tmp.c0_p[x_idx]
            + tmp.c0_p[y_idx];
    }

    fn insertion_prob(tree_length: f64, b: f64, mu: f64) -> f64 {
        b / (tree_length + 1.0 / mu)
    }

    fn survival_prob(mu: f64, b: f64) -> f64 {
        if b == 0.0 {
            return 1.0;
        }
        (1.0 - (-mu * b).exp()) / (mu * b)
    }
}

fn log_factorial(n: usize) -> f64 {
    (1..n + 1).map(|i| (i as f64).ln()).sum()
}

#[cfg(test)]
mod tests;
