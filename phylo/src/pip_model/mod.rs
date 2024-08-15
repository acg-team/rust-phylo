use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Mul;
use std::vec;

use nalgebra::{DMatrix, DVector};

use crate::evolutionary_models::{EvoModelInfo, EvolutionaryModel};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::dna_models::DNASubstModel;
use crate::substitution_models::protein_models::ProteinSubstModel;
use crate::substitution_models::{FreqVector, SubstMatrix, SubstitutionModel};
use crate::tree::NodeIdx::{self, Internal as Int, Leaf};
use crate::Result;

mod pip_parameters;
pub use pip_parameters::*;

pub struct PIPModel<SubstModel: SubstitutionModel> {
    pub(crate) q: SubstMatrix,
    pub subst_model: SubstModel,
    pub(crate) index: [usize; 255],
    pub params: PIPParams<SubstModel>,
}

pub type PIPDNAModel = PIPModel<DNASubstModel>;
pub type PIPProteinModel = PIPModel<ProteinSubstModel>;

impl<SubstModel: SubstitutionModel> PIPModel<SubstModel> {
    fn make_pip_q(
        index: [usize; 255],
        subst_model: &SubstModel,
        mu: f64,
    ) -> ([usize; 255], SubstMatrix, FreqVector) {
        let mut index = index;
        index[b'-' as usize] = SubstModel::N;
        let mut q = SubstitutionModel::q(subst_model)
            .clone()
            .insert_column(SubstModel::N, mu)
            .insert_row(SubstModel::N, 0.0);
        q.fill_diagonal(0.0);
        for i in 0..(SubstModel::N + 1) {
            q[(i, i)] = -q.row(i).sum();
        }
        let pi = SubstitutionModel::freqs(subst_model)
            .clone()
            .insert_row(SubstModel::N, 0.0);
        (index, q, pi)
    }
}

impl<SubstModel: SubstitutionModel> PIPModel<SubstModel>
where
    SubstModel: Clone,
    SubstModel::Params: Clone,
    SubstModel::ModelType: Clone,
{
    pub fn create(params: &PIPParams<SubstModel>) -> PIPModel<SubstModel> {
        let mut subst_model = SubstModel::create(&params.subst_params);
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
            subst_model,
        }
    }
}

impl<SubstModel: SubstitutionModel> EvolutionaryModel for PIPModel<SubstModel>
where
    SubstModel: Clone,
    SubstModel::Params: Clone,
    SubstModel::ModelType: Clone,
{
    type ModelType = SubstModel::ModelType;
    type Params = PIPParams<SubstModel>;

    fn new(model: Self::ModelType, params: &[f64]) -> Result<Self>
    where
        Self: Sized,
    {
        let params = PIPParams::<SubstModel>::new(&model, params)?;
        Ok(Self::create(&params))
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

    fn index(&self) -> &[usize; 255] {
        &self.index
    }

    fn params(&self) -> &PIPParams<SubstModel> {
        &self.params
    }
}

#[derive(Debug)]
pub struct PIPModelInfo<SubstModel: SubstitutionModel> {
    phantom: PhantomData<SubstModel>,
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

impl<SubstModel: SubstitutionModel + Clone> EvoModelInfo for PIPModelInfo<SubstModel>
where
    SubstModel::Params: Clone,
    SubstModel::ModelType: Clone,
{
    type Model = PIPModel<SubstModel>;

    fn new(info: &PhyloInfo, model: &Self::Model) -> Result<Self> {
        let node_count = info.tree.len();
        let msa_length = info.msa_length();
        let mut leaf_seq_info: HashMap<String, DMatrix<f64>> = HashMap::new();
        for node in info.tree.leaves() {
            let alignment_map = info.msa.leaf_map(&node.idx);
            let leaf_encoding = info.leaf_encoding.get(&node.id).unwrap();
            let mut leaf_seq_w_gaps = DMatrix::<f64>::zeros(SubstModel::N + 1, msa_length);
            for (i, mut site_info) in leaf_seq_w_gaps.column_iter_mut().enumerate() {
                if let Some(c) = alignment_map[i] {
                    let encoding = leaf_encoding.column(c).insert_row(SubstModel::N, 0.0);
                    site_info.copy_from(&encoding);
                    site_info.component_mul_assign(model.freqs());
                } else {
                    site_info.fill_row(SubstModel::N, 1.0);
                }
                site_info.scale_mut((1.0) / site_info.sum());
            }
            leaf_seq_info.insert(node.id.clone(), leaf_seq_w_gaps);
        }
        let ftilde = info
            .tree
            .iter()
            .map(|node| match node.idx {
                Int(_) => DMatrix::<f64>::zeros(SubstModel::N + 1, msa_length),
                Leaf(_) => leaf_seq_info
                    .get(info.tree.node_id(&node.idx))
                    .unwrap()
                    .clone(),
            })
            .collect();

        Ok(PIPModelInfo::<SubstModel> {
            tree_length: info.tree.all_branch_lengths().iter().sum(),
            ftilde,
            ins_probs: vec![0.0; node_count],
            surv_probs: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            p: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_ftilde: vec![DMatrix::<f64>::zeros(SubstModel::N + 1, 1); node_count],
            c0_f: vec![0.0; node_count],
            c0_p: vec![0.0; node_count],
            branches: info.tree.iter().map(|n| n.blen).collect(),
            anc: vec![DMatrix::<f64>::zeros(msa_length, 3); node_count],
            valid: vec![false; node_count],
            models: vec![SubstMatrix::zeros(SubstModel::N + 1, SubstModel::N + 1); node_count],
            models_valid: vec![false; node_count],
            phantom: PhantomData,
        })
    }

    fn reset(&mut self) {
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

#[derive(Clone)]
pub struct PIPCost<'a, SubstModel: SubstitutionModel> {
    pub(crate) info: PhyloInfo,
    pub(crate) model: &'a PIPModel<SubstModel>,
}

impl<'a, SubstModel: SubstitutionModel> LikelihoodCostFunction for PIPCost<'a, SubstModel>
where
    SubstModel: Clone,
    SubstModel::Params: Clone,
    SubstModel::ModelType: Clone,
{
    type Model = PIPModel<SubstModel>;
    type Info = PIPModelInfo<SubstModel>;

    fn logl(&self) -> f64 {
        self.logl().0
    }
}

impl<'a, SubstModel: SubstitutionModel> PIPCost<'a, SubstModel>
where
    SubstModel: Clone,
    SubstModel::ModelType: Clone,
    SubstModel::Params: Clone,
{
    pub(crate) fn logl(&self) -> (f64, PIPModelInfo<SubstModel>) {
        let mut tmp_info = PIPModelInfo::<SubstModel>::new(&self.info, self.model).unwrap();
        (self.logl_with_tmp(&mut tmp_info), tmp_info)
    }

    fn logl_with_tmp(&self, tmp: &mut PIPModelInfo<SubstModel>) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Int(_) => {
                    if self.info.tree.root == *node_idx {
                        self.set_root(node_idx, self.model, tmp);
                    }
                    self.set_internal(node_idx, self.model, tmp);
                }
                Leaf(_) => {
                    self.set_leaf(node_idx, self.model, tmp);
                }
            };
        }
        let root_idx = usize::from(&self.info.tree.root);
        let msa_length = tmp.ftilde[0].ncols();
        let nu = self.model.params.lambda * (tmp.tree_length + 1.0 / self.model.params.mu);
        let ln_phi = nu.ln() * msa_length as f64 + (tmp.c0_p[root_idx] - 1.0) * nu
            - (log_factorial(msa_length));
        tmp.p[root_idx].map(|x| x.ln()).sum() + ln_phi
    }

    fn set_root(
        &self,
        idx: &NodeIdx,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        self.set_model(idx, model, tmp);
        let idx_usize = usize::from(idx);
        if !tmp.valid[idx_usize] {
            let mu = model.params.mu;
            tmp.surv_probs[idx_usize] = 1.0;
            tmp.ins_probs[idx_usize] = Self::insertion_prob(tmp.tree_length, 1.0 / mu, mu);
            self.set_ftilde(idx, model, tmp);
            self.set_ancestors(idx, tmp);
            tmp.anc[idx_usize].fill_column(0, 1.0);
            self.set_p(idx, tmp);
            self.set_c0(idx, model, tmp);
            tmp.valid[idx_usize] = true;
        }
    }

    fn set_internal(
        &self,
        idx: &NodeIdx,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        self.set_model(idx, model, tmp);
        let us_idx = usize::from(idx);
        if !tmp.valid[us_idx] {
            let mu = model.params.mu;
            let b = tmp.branches[us_idx];
            tmp.surv_probs[us_idx] = Self::survival_prob(mu, b);
            tmp.ins_probs[us_idx] = Self::insertion_prob(tmp.tree_length, b, mu);
            self.set_ftilde(idx, model, tmp);
            self.set_ancestors(idx, tmp);
            self.set_p(idx, tmp);
            self.set_c0(idx, model, tmp);
            tmp.valid[us_idx] = true;
        }
    }

    fn set_leaf(
        &self,
        idx: &NodeIdx,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let us_idx = usize::from(idx);
        if !tmp.valid[us_idx] {
            self.set_model(idx, model, tmp);
            let mu = model.params.mu;
            let b = tmp.branches[us_idx];
            tmp.surv_probs[us_idx] = Self::survival_prob(mu, b);
            tmp.ins_probs[us_idx] = Self::insertion_prob(tmp.tree_length, b, mu);
            for (i, c) in self.info.msa.leaf_map(idx).iter().enumerate() {
                if c.is_some() {
                    tmp.anc[us_idx][(i, 0)] = 1.0;
                }
            }
            tmp.f[us_idx] = tmp.ftilde[us_idx]
                .tr_mul(model.freqs())
                .component_mul(&tmp.anc[us_idx].column(0));
            tmp.p[us_idx] = tmp.f[us_idx].clone() * tmp.surv_probs[us_idx] * tmp.ins_probs[us_idx];
            tmp.c0_ftilde[us_idx][SubstModel::N] = 1.0;
            tmp.c0_f[us_idx] = 0.0;
            tmp.c0_p[us_idx] = (1.0 - tmp.surv_probs[us_idx]) * tmp.ins_probs[us_idx];
            tmp.valid[us_idx] = true;
        }
    }

    fn set_p(&self, idx: &NodeIdx, tmp: &mut PIPModelInfo<SubstModel>) {
        let node = self.info.tree.node(idx);
        let idx = usize::from(idx);
        tmp.p[idx] = tmp.f[idx].clone().component_mul(&tmp.anc[idx].column(0))
            * tmp.surv_probs[idx]
            * tmp.ins_probs[idx];
        let x_p = tmp.p[usize::from(&node.children[0])].clone();
        let y_p = tmp.p[usize::from(&node.children[1])].clone();
        tmp.p[idx] +=
            tmp.anc[idx].column(1).component_mul(&x_p) + tmp.anc[idx].column(2).component_mul(&y_p);
    }

    fn set_ftilde(
        &self,
        idx: &NodeIdx,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let node = self.info.tree.node(idx);
        let idx = usize::from(idx);
        let x_idx = usize::from(&node.children[0]);
        let y_idx = usize::from(&node.children[1]);
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = tmp.ftilde[idx].tr_mul(model.freqs());
    }

    fn set_model(
        &self,
        idx: &NodeIdx,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let idx = usize::from(idx);
        if !tmp.models_valid[idx] {
            tmp.models[idx] = model.p(tmp.branches[idx]);
            tmp.models_valid[idx] = true;
        }
    }

    fn set_ancestors(&self, idx: &NodeIdx, tmp: &mut PIPModelInfo<SubstModel>) {
        let node = self.info.tree.node(idx);
        let idx = usize::from(idx);
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

    fn set_c0(
        &self,
        idx: &NodeIdx,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let node = self.info.tree.node(idx);
        let x_idx = usize::from(&node.children[0]);
        let y_idx = usize::from(&node.children[1]);
        let idx = usize::from(idx);
        tmp.c0_ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.c0_ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.c0_ftilde[y_idx]));
        tmp.c0_f[idx] = tmp.c0_ftilde[idx].tr_mul(model.freqs())[0];
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
    (1..n + 1).map(|i| (i as f64).ln()).sum::<f64>()
}

#[cfg(test)]
mod tests;
