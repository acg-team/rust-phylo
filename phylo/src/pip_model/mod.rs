use std::collections::HashMap;
use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::Mul;
use std::vec;

use nalgebra::{DMatrix, DVector};

use crate::alignment::Mapping;
use crate::evolutionary_models::EvoModel;
use crate::likelihood::PhyloCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{
    DNASubstModel, FreqVector, ProteinSubstModel, SubstMatrix, SubstitutionModel,
};
use crate::tree::{
    Node,
    NodeIdx::{Internal as Int, Leaf},
};
use crate::Result;

mod pip_parameters;
pub use pip_parameters::*;

#[derive(Clone)]
pub struct PIPModel<SM: SubstitutionModel + Clone>
where
    SM::ModelType: Clone,
{
    pub(crate) q: SubstMatrix,
    pub(crate) index: [usize; 255],
    pub params: PIPParams<SM>,
}

pub type PIPDNAModel = PIPModel<DNASubstModel>;
pub type PIPProteinModel = PIPModel<ProteinSubstModel>;

impl<SM: SubstitutionModel + Clone> EvoModel for PIPModel<SM>
where
    SM::ModelType: Clone + Display,
    PIPParameter: Into<SM::Parameter>,
    SM::Parameter: Into<PIPParameter>,
{
    type Parameter = PIPParameter;
    type ModelType = SM::ModelType;
    const N: usize = SM::N + 1;

    fn new(model: SM::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = PIPParams::<SM>::new(model, params)?;
        Ok(Self::create(&params))
    }

    fn model_type(&self) -> &Self::ModelType {
        &self.params.model_type
    }

    fn description(&self) -> String {
        format!("PIP with {}", self.params.model_type)
    }

    fn p(&self, time: f64) -> SubstMatrix {
        (self.q.clone() * time).exp()
    }

    fn q(&self) -> &SubstMatrix {
        &self.q
    }

    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(self.index[i as usize], self.index[j as usize])]
    }

    fn freqs(&self) -> &FreqVector {
        &self.params.pi
    }

    fn set_freqs(&mut self, pi: FreqVector) {
        self.params.set_freqs(pi);
        self.update();
    }

    fn index(&self) -> &[usize; 255] {
        &self.index
    }

    fn model_parameters(&self) -> Vec<PIPParameter> {
        SubstitutionModel::model_parameters(&self.params.subst_model)
            .into_iter()
            .map(|param| param.into())
            .chain(vec![PIPParameter::Mu, PIPParameter::Lambda])
            .collect()
    }

    fn param(&self, param_name: &PIPParameter) -> f64 {
        self.params.param(param_name)
    }

    fn set_param(&mut self, param_name: &PIPParameter, value: f64) {
        self.params.set_param(param_name, value);
        self.update();
    }
}

impl<SM: SubstitutionModel + Display + Clone> Display for PIPModel<SM>
where
    SM::ModelType: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.params)
    }
}

impl<SM: SubstitutionModel + Clone> PIPModel<SM>
where
    SM::ModelType: Clone,
    PIPParams<SM>: Clone,
    PIPParameter: Into<SM::Parameter>,
    SM::Parameter: Into<PIPParameter>,
{
    pub(crate) fn create(params: &PIPParams<SM>) -> PIPModel<SM> {
        let mut subst_model = params.subst_model.clone();
        subst_model.normalise();
        let (index, q, _) = Self::make_pip_q(
            *SubstitutionModel::index(&subst_model),
            &subst_model,
            params.mu,
        );
        PIPModel {
            index,
            params: params.clone(),
            q,
        }
    }

    fn update(&mut self) {
        let (index, q, pi) = Self::make_pip_q(
            *SubstitutionModel::index(&self.params.subst_model),
            &self.params.subst_model,
            self.params.mu,
        );
        self.index = index;
        self.q = q;
        self.params.pi = pi;
    }

    fn make_pip_q(
        index: [usize; 255],
        subst_model: &SM,
        mu: f64,
    ) -> ([usize; 255], SubstMatrix, FreqVector) {
        let n = SM::N;
        let mut index = index;
        index[b'-' as usize] = n;
        let mut q = subst_model
            .q()
            .clone()
            .insert_column(n, mu)
            .insert_row(n, 0.0);
        q.fill_diagonal(0.0);
        for i in 0..(n + 1) {
            q[(i, i)] = -q.row(i).sum();
        }
        let pi = subst_model.freqs().clone().insert_row(n, 0.0);
        (index, q, pi)
    }
}

#[derive(Debug)]
pub struct PIPModelInfo<SM: SubstitutionModel> {
    phantom: PhantomData<SM>,
    tree_length: f64,
    ins_probs: Vec<f64>,
    surv_probs: Vec<f64>,
    anc: Vec<DMatrix<f64>>,
    branches: Vec<f64>,
    ftilde: Vec<DMatrix<f64>>,
    f: Vec<DVector<f64>>,
    p: Vec<DVector<f64>>,
    c0_ftilde: Vec<DMatrix<f64>>,
    c0_f: Vec<f64>,
    c0_p: Vec<f64>,
    valid: Vec<bool>,
    models: Vec<SubstMatrix>,
    models_valid: Vec<bool>,
}

impl<SM: SubstitutionModel + Clone> PIPModelInfo<SM>
where
    SM::ModelType: Clone,
    PIPModel<SM>: EvoModel,
{
    pub fn new(info: &PhyloInfo, model: &PIPModel<SM>) -> Result<Self> {
        let n = PIPModel::<SM>::N;
        let node_count = info.tree.len();
        let msa_length = info.msa_length();
        let mut leaf_seq_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(n, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = leaf_encoding.column(c).insert_row(n - 1, 0.0);
                    site_info.copy_from(&encoding);
                    site_info.component_mul_assign(model.freqs());
                } else {
                    site_info.fill_row(n - 1, 1.0);
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_seq_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        let ftilde = info
            .tree
            .iter()
            .map(|node| match node.idx {
                Int(_) => DMatrix::<f64>::zeros(n, msa_length),
                Leaf(_) => leaf_seq_info
                    .get(info.tree.node_id(&node.idx))
                    .unwrap()
                    .clone(),
            })
            .collect();

        Ok(PIPModelInfo::<SM> {
            tree_length: info.tree.iter().map(|n| n.blen).sum(),
            ftilde,
            ins_probs: vec![0.0; node_count],
            surv_probs: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            p: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_ftilde: vec![DMatrix::<f64>::zeros(n, 1); node_count],
            c0_f: vec![0.0; node_count],
            c0_p: vec![0.0; node_count],
            branches: info.tree.iter().map(|n| n.blen).collect(),
            anc: vec![DMatrix::<f64>::zeros(msa_length, 3); node_count],
            valid: vec![false; node_count],
            models: vec![SubstMatrix::zeros(n, n); node_count],
            models_valid: vec![false; node_count],
            phantom: PhantomData,
        })
    }

    pub fn reset(&mut self) {
        self.ins_probs.fill(0.0);
        self.surv_probs.fill(0.0);
        self.anc.iter_mut().for_each(|x| x.fill(0.0));
        self.ftilde.iter_mut().for_each(|x| x.fill(0.0));
        self.f.iter_mut().for_each(|x| x.fill(0.0));
        self.p.iter_mut().for_each(|x| x.fill(0.0));
        self.c0_ftilde.iter_mut().for_each(|x| x.fill(0.0));
        self.c0_f.fill(0.0);
        self.c0_p.fill(0.0);
        self.valid.fill(false);
        self.models.iter_mut().for_each(|x| x.fill(0.0));
        self.models_valid.fill(false);
    }
}

impl<SM: SubstitutionModel + Clone> PhyloCostFunction for PIPModel<SM>
where
    SM::ModelType: Clone,
    PIPModel<SM>: EvoModel,
{
    fn cost(&self, info: &PhyloInfo) -> f64 {
        self.logl(info).0
    }

    fn reset(&self) {
        // TODO: Implement reset once the tmp storage is used properly
    }
}

impl<SM: SubstitutionModel + Clone> PIPModel<SM>
where
    SM::ModelType: Clone,
    PIPModel<SM>: EvoModel,
{
    fn logl(&self, info: &PhyloInfo) -> (f64, PIPModelInfo<SM>) {
        let mut tmp_info = PIPModelInfo::<SM>::new(info, self).unwrap();
        (self.logl_with_tmp(info, &mut tmp_info), tmp_info)
    }

    fn logl_with_tmp(&self, info: &PhyloInfo, tmp: &mut PIPModelInfo<SM>) -> f64 {
        tmp.tree_length = info.tree.height;
        for node_idx in info.tree.postorder() {
            match node_idx {
                Int(_) => {
                    if info.tree.root == *node_idx {
                        self.set_root(info.tree.node(node_idx), tmp);
                    }
                    self.set_internal(info.tree.node(node_idx), tmp);
                }
                Leaf(_) => {
                    self.set_leaf(info.tree.node(node_idx), info.msa.leaf_map(node_idx), tmp);
                }
            };
        }
        let root_idx = usize::from(&info.tree.root);
        let msa_length = tmp.ftilde[0].ncols();
        let nu = self.params.lambda * (tmp.tree_length + 1.0 / self.params.mu);
        let ln_phi = nu.ln() * msa_length as f64 + (tmp.c0_p[root_idx] - 1.0) * nu
            - (log_factorial(msa_length));
        tmp.p[root_idx].map(|x| x.ln()).sum() + ln_phi
    }

    fn set_root(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        self.set_model(node, tmp);
        let idx = usize::from(node.idx);
        if !tmp.valid[idx] {
            let mu = self.params.mu;
            tmp.surv_probs[idx] = 1.0;
            tmp.ins_probs[idx] = Self::insertion_prob(tmp.tree_length, 1.0 / mu, mu);
            self.set_ftilde(node, tmp);
            self.set_ancestors(node, tmp);
            tmp.anc[idx].fill_column(0, 1.0);
            self.set_p(node, tmp);
            self.set_c0(node, tmp);
            tmp.valid[idx] = true;
        }
    }

    fn set_internal(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        self.set_model(node, tmp);
        let us_idx = usize::from(node.idx);
        if !tmp.valid[us_idx] {
            let mu = self.params.mu;
            let b = tmp.branches[us_idx];
            tmp.surv_probs[us_idx] = Self::survival_prob(mu, b);
            tmp.ins_probs[us_idx] = Self::insertion_prob(tmp.tree_length, b, mu);
            self.set_ftilde(node, tmp);
            self.set_ancestors(node, tmp);
            self.set_p(node, tmp);
            self.set_c0(node, tmp);
            tmp.valid[us_idx] = true;
        }
    }

    fn set_leaf(&self, node: &Node, leaf_map: &Mapping, tmp: &mut PIPModelInfo<SM>) {
        let idx = usize::from(node.idx);
        if !tmp.valid[idx] {
            self.set_model(node, tmp);
            let mu = self.params.mu;
            let b = tmp.branches[idx];
            tmp.surv_probs[idx] = Self::survival_prob(mu, b);
            tmp.ins_probs[idx] = Self::insertion_prob(tmp.tree_length, b, mu);
            for (i, c) in leaf_map.iter().enumerate() {
                if c.is_some() {
                    tmp.anc[idx][(i, 0)] = 1.0;
                }
            }
            tmp.f[idx] = tmp.ftilde[idx]
                .tr_mul(self.freqs())
                .component_mul(&tmp.anc[idx].column(0));
            tmp.p[idx] = tmp.f[idx].clone() * tmp.surv_probs[idx] * tmp.ins_probs[idx];
            tmp.c0_ftilde[idx][PIPModel::<SM>::N - 1] = 1.0;
            tmp.c0_f[idx] = 0.0;
            tmp.c0_p[idx] = (1.0 - tmp.surv_probs[idx]) * tmp.ins_probs[idx];
            tmp.valid[idx] = true;
        }
    }

    fn set_p(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        let idx = usize::from(node.idx);
        tmp.p[idx] = tmp.f[idx].clone().component_mul(&tmp.anc[idx].column(0))
            * tmp.surv_probs[idx]
            * tmp.ins_probs[idx];
        let x_p = tmp.p[usize::from(&node.children[0])].clone();
        let y_p = tmp.p[usize::from(&node.children[1])].clone();
        tmp.p[idx] +=
            tmp.anc[idx].column(1).component_mul(&x_p) + tmp.anc[idx].column(2).component_mul(&y_p);
    }

    fn set_ftilde(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        let idx = usize::from(node.idx);
        let x_idx = usize::from(&node.children[0]);
        let y_idx = usize::from(&node.children[1]);
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = tmp.ftilde[idx].tr_mul(self.freqs());
    }

    fn set_model(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        let idx = usize::from(node.idx);
        if !tmp.models_valid[idx] {
            tmp.models[idx] = self.p(tmp.branches[idx]);
            tmp.models_valid[idx] = true;
        }
    }

    fn set_ancestors(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        let idx = usize::from(node.idx);
        let x_anc = tmp.anc[usize::from(&node.children[0])].clone();
        let y_anc = tmp.anc[usize::from(&node.children[1])].clone();
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

    fn set_c0(&self, node: &Node, tmp: &mut PIPModelInfo<SM>) {
        let x_idx = usize::from(&node.children[0]);
        let y_idx = usize::from(&node.children[1]);
        let idx = usize::from(node.idx);
        tmp.c0_ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.c0_ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.c0_ftilde[y_idx]));
        tmp.c0_f[idx] = tmp.c0_ftilde[idx].tr_mul(self.freqs())[0];
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
