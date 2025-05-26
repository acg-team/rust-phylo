use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::iter;
use std::marker::PhantomData;
use std::ops::Mul;
use std::vec;

use anyhow::bail;
use fixedbitset::FixedBitSet;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use log::warn;
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

// (2.0 * PI).ln() / 2.0;
pub static SHIFT: f64 = 0.9189385332046727;

lazy_static! {
    pub static ref MINLOGPROB: f64 = (f64::MIN_POSITIVE).ln();
}

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

impl<Q: QMatrix> PIPModel<Q> {
    fn lambda(&self) -> f64 {
        self.params[0]
    }

    fn mu(&self) -> f64 {
        self.params[1]
    }
}

impl<Q: QMatrix + QMatrixMaker> PIPModel<Q> {
    pub fn new(frequencies: &[f64], params: &[f64]) -> Self {
        let mut params = params.to_vec();
        if params.len() < 2 {
            warn!("Too few values provided for PIP, 2 values required, lambda and mu.");
            warn!("Falling back to default values.");
            params.extend(iter::repeat_n(1.5, 2 - params.len()));
        }
        let mu = params[1];

        let subst_q = Q::create(frequencies, &params[2..]);
        let n = subst_q.n();
        let freqs = subst_q.freqs().clone().insert_row(n, 0.0);
        let mut q = SubstMatrix::zeros(n + 1, n + 1);
        pip_q(&mut q, subst_q.q(), mu);
        PIPModel {
            subst_q,
            q,
            freqs,
            params: params.to_vec(),
        }
    }
}

impl<Q: QMatrix> EvoModel for PIPModel<Q> {
    fn p(&self, time: f64) -> SubstMatrix {
        (self.q() * time).exp()
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
        self.subst_q.set_freqs(pi);
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
pub struct PIPModelCacheEntry {
    surv_ins_weights: f64,
    anc: DMatrix<f64>,
    ftilde: DMatrix<f64>,
    f: DVector<f64>,
    pnu: DVector<f64>,
    c0_f1: f64,
    c0_pnu: f64,
    models: SubstMatrix,
}

impl PIPModelCacheEntry {
    fn make_default(n: usize, msa_length: usize) -> Self {
        Self {
            ftilde: DMatrix::<f64>::zeros(n, msa_length),
            surv_ins_weights: 0.0,
            f: DVector::<f64>::zeros(msa_length),
            pnu: DVector::<f64>::zeros(msa_length),
            c0_f1: 0.0,
            c0_pnu: 0.0,
            anc: DMatrix::<f64>::zeros(msa_length, 3),
            models: SubstMatrix::zeros(n, n),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PIPModelInfo<Q: QMatrix> {
    phantom: PhantomData<Q>,
    cache: Vec<PIPModelCacheEntry>,
    valid: FixedBitSet,
    models_valid: FixedBitSet,
    leaf_seq_info: HashMap<usize, DMatrix<f64>>,
}

impl<Q: QMatrix> PIPModelInfo<Q> {
    pub fn new(info: &PhyloInfo, model: &PIPModel<Q>) -> Result<Self> {
        let n = model.q().nrows();
        let node_count = info.tree.len();
        let msa_length = info.msa.len();
        let mut leaf_seq_info = HashMap::with_capacity(info.tree.leaves().len());
        for node in info.tree.leaves() {
            let seq = info.msa.seqs.record_by_id(&node.id).seq().to_vec();

            let alignment_map = info.msa.leaf_map(&node.idx);
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(n, msa_length);

            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = info
                        .msa
                        .alphabet()
                        .char_encoding(seq[c])
                        .clone()
                        .insert_row(n - 1, 0.0);

                    site_info.copy_from(&encoding);
                } else {
                    site_info.fill_row(n - 1, 1.0);
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_seq_info.insert(usize::from(node.idx), leaf_seq_w_gaps);
        }
        Ok(PIPModelInfo {
            phantom: PhantomData,
            cache: vec![PIPModelCacheEntry::make_default(n, msa_length); node_count],
            valid: FixedBitSet::with_capacity(node_count),
            models_valid: FixedBitSet::with_capacity(node_count),
            leaf_seq_info,
        })
    }
}

pub struct PIPCostBuilder<Q: QMatrix> {
    model: PIPModel<Q>,
    info: PhyloInfo,
}

impl<Q: QMatrix> PIPCostBuilder<Q> {
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

impl<Q: QMatrix> TreeSearchCost for PIPCost<Q> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]) {
        self.info.tree = tree;
        if dirty_nodes.is_empty() {
            self.tmp.borrow_mut().valid.clear();
            self.tmp.borrow_mut().models_valid.clear();
            return;
        }
        for node_idx in dirty_nodes {
            self.tmp.borrow_mut().valid.remove(usize::from(node_idx));
            self.tmp
                .borrow_mut()
                .models_valid
                .remove(usize::from(node_idx));
        }
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

impl<Q: QMatrix> ModelSearchCost for PIPCost<Q> {
    fn cost(&self) -> f64 {
        self.logl()
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.model.set_param(param, value);
        self.tmp.borrow_mut().models_valid.clear();
    }

    fn params(&self) -> &[f64] {
        self.model.params()
    }

    fn set_freqs(&mut self, freqs: FreqVector) {
        self.model.set_freqs(freqs);
        self.tmp.borrow_mut().models_valid.clear();
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

#[derive(Debug)]
struct PIPCostNodeCache<'a> {
    children: [&'a mut PIPModelCacheEntry; 2],
    this: &'a mut PIPModelCacheEntry,
}

impl<'a> PIPCostNodeCache<'a> {
    fn new(
        children: [&'a mut PIPModelCacheEntry; 2],
        this: &'a mut PIPModelCacheEntry,
    ) -> PIPCostNodeCache<'a> {
        Self { children, this }
    }
    fn left_right_this_mut(&mut self) -> [&mut PIPModelCacheEntry; 3] {
        let [left, right] = &mut self.children;
        [left, right, self.this]
    }
    fn this_mut(&mut self) -> &mut PIPModelCacheEntry {
        self.this
    }
}

impl<Q: QMatrix> PIPCost<Q> {
    fn logl(&self) -> f64 {
        for node_idx in self.tree().postorder() {
            let number_node_idx = usize::from(node_idx);
            let mut tmp = self.tmp.borrow_mut();
            let PIPModelInfo {
                cache,
                valid: valid_cache_entries,
                models_valid: valid_model_cache_entries,
                leaf_seq_info,
                ..
            } = &mut *tmp;
            if self.tree().dirty[number_node_idx] || !valid_model_cache_entries[number_node_idx] {
                self.set_model(
                    self.tree().nodes[number_node_idx].blen,
                    &mut cache[number_node_idx],
                );
                valid_model_cache_entries.insert(number_node_idx);
                valid_cache_entries.remove(number_node_idx);
            }

            match node_idx {
                Int(_) => {
                    let children = &self.tree().nodes[number_node_idx].children;
                    let children = [children[0], children[1]].map(usize::from);
                    let children_blen = children.map(|child_idx| self.tree().nodes[child_idx].blen);

                    let [left, right, this] = cache
                        .get_disjoint_mut([children[0], children[1], usize::from(node_idx)])
                        .expect("children and parent should be distinct");
                    let indices = &mut PIPCostNodeCache::new([left, right], this);

                    if !valid_cache_entries[number_node_idx] {
                        if self.tree().root == *node_idx {
                            self.set_root(children_blen, indices);
                        } else {
                            self.set_internal(
                                self.tree().nodes[number_node_idx].blen,
                                children_blen,
                                indices,
                            );
                            let parent_idx = self.tree().nodes[number_node_idx]
                                .parent
                                .expect("all internal nodes have a parent");
                            valid_cache_entries.remove(usize::from(parent_idx));
                        }
                        valid_cache_entries.insert(number_node_idx)
                    }
                }
                Leaf(_) => {
                    self.set_leaf(
                        self.tree().nodes[number_node_idx].blen,
                        &mut cache[number_node_idx],
                        leaf_seq_info
                            .get(&number_node_idx)
                            .expect("leaf should have sequence info"),
                        self.info.msa.leaf_map(node_idx),
                    );
                }
            };
        }

        let tmp = self.tmp.borrow();
        let root_idx = usize::from(&self.info.tree.root);
        let msa_length = self.info.msa.len();

        // In certain scenarios (e.g. a completely unrelated sequence, see data/p105.msa.fa)
        // individual column probabilities become too close to 0.0 (become subnormal)
        // and the log likelihood becomes -Inf. This is mathematically reasonable, but during branch
        // length optimisation BrentOpt cannot handle it and proposes NaN branch lengths.
        // This is a workaround that sets the probability to the smallest posible positive float,
        // which is equivalent to restricting the log likelihood to f64::MIN.
        let root = &tmp.cache[root_idx];
        root.pnu
            .map(|x| {
                if x == 0.0 || x.is_subnormal() {
                    *MINLOGPROB
                } else {
                    x.ln()
                }
            })
            .sum()
            + root.c0_pnu
            - log_factorial_shifted(msa_length)
    }

    // TODO: MERBUG try #[cold]
    fn set_root<'a, 'b: 'a>(&self, children_blen: [f64; 2], indices: &'a mut PIPCostNodeCache<'b>) {
        let this = indices.this_mut();
        this.surv_ins_weights = self.model.lambda() / self.model.mu();
        this.anc.fill_column(0, 1.0);

        self.set_ftilde(indices);
        self.set_ancestors(indices);
        self.set_pnu(indices);
        self.set_c0(children_blen, indices);
    }

    fn set_internal<'a, 'b: 'a>(
        &self,
        node_blen: f64,
        children_blen: [f64; 2],
        indices: &'a mut PIPCostNodeCache<'b>,
    ) {
        indices.this_mut().surv_ins_weights =
            Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node_blen);

        self.set_ftilde(indices);
        self.set_ancestors(indices);
        self.set_pnu(indices);
        self.set_c0(children_blen, indices);
    }

    fn set_leaf(
        &self,
        node_blen: f64,
        PIPModelCacheEntry {
            f,
            ftilde,
            pnu,
            c0_f1,
            c0_pnu,
            surv_ins_weights,
            anc,
            ..
        }: &mut PIPModelCacheEntry,
        leaf_seq_info: &DMatrix<f64>,
        leaf_map: &Mapping,
    ) {
        *surv_ins_weights =
            Self::survival_insertion_weight(self.model.lambda(), self.model.mu(), node_blen);
        for (i, c) in leaf_map.iter().enumerate() {
            if c.is_some() {
                anc[(i, 0)] = 1.0;
            }
        }

        leaf_seq_info.clone_into(ftilde);

        *f = ftilde
            .tr_mul(self.model.freqs())
            .component_mul(&anc.column(0));

        *pnu = &*f * *surv_ins_weights;
        *c0_f1 = -1.0;
        *c0_pnu = -*surv_ins_weights;
    }

    /// Dependent:
    /// - tmp.pnu of both children
    /// - tmp.anc of this node
    /// - tmp.f of this node
    /// - tmp.surv_ins_weights of this node
    ///
    /// Modifies:
    /// - tmp.pnu of this node
    fn set_pnu<'a, 'b: 'a>(&self, indices: &'a mut PIPCostNodeCache<'b>) {
        let [left, right, this] = indices.left_right_this_mut();

        this.pnu = this.f.component_mul(&this.anc.column(0)) * this.surv_ins_weights;
        this.pnu += this.anc.column(1).component_mul(&left.pnu)
            + this.anc.column(2).component_mul(&right.pnu);
    }

    /// Dependent:
    /// - tmp.models for both children
    /// - tmp.ftilde for both children
    ///
    /// Modifies:
    /// - tmp.ftilde for this node
    /// - tmp.f for this node
    fn set_ftilde<'a, 'b: 'a>(&self, indices: &'a mut PIPCostNodeCache<'b>) {
        let [left, right, this] = indices.left_right_this_mut();
        this.ftilde = (&left.models)
            .mul(&left.ftilde)
            .component_mul(&(&right.models).mul(&right.ftilde));
        this.f = this.ftilde.tr_mul(self.model.freqs());
    }

    fn set_model(
        &self,
        node_blen: f64,
        PIPModelCacheEntry { models, .. }: &mut PIPModelCacheEntry,
    ) {
        *models = self.model.p(node_blen);
    }

    /// Dependent:
    /// - tmp.anc of both children
    ///
    /// Modifies:
    /// - tmp.anc of this node
    fn set_ancestors<'a, 'b: 'a>(&self, indices: &'a mut PIPCostNodeCache<'b>) {
        let [left, right, this] = indices.left_right_this_mut();
        this.anc.set_column(1, &left.anc.column(0));
        this.anc.set_column(2, &right.anc.column(0));
        this.anc
            .set_column(0, &(left.anc.column(0) + right.anc.column(0)));
        for i in 0..this.anc.nrows() {
            debug_assert!((0.0..=2.0).contains(&this.anc[(i, 0)]));
            if this.anc[(i, 0)] == 2.0 {
                this.anc.fill_row(i, 0.0);
                this.anc[(i, 0)] = 1.0;
            }
        }
    }

    /// Dependent:
    /// - blen of both children
    /// - tmp.c0_f1 of both children
    /// - tmp.c0_pnu of both children
    ///
    /// Modifies:
    /// - tmp.c0_f1 of this node
    /// - tmp.c0_pnu of this node
    fn set_c0<'a, 'b: 'a>(
        &self,
        [left_blen, right_blen]: [f64; 2],
        indices: &'a mut PIPCostNodeCache<'b>,
    ) {
        let [left, right, this] = indices.left_right_this_mut();

        let mu = self.model.mu();
        this.c0_f1 = (1.0 + (-mu * left_blen).exp() * left.c0_f1)
            * (1.0 + (-mu * right_blen).exp() * right.c0_f1)
            - 1.0;

        this.c0_pnu = this.surv_ins_weights * this.c0_f1 + left.c0_pnu + right.c0_pnu;
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
