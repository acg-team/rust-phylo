use std::marker::PhantomData;
use std::ops::Mul;
use std::vec;

use anyhow::bail;
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
        let mut q = SubstitutionModel::get_q(subst_model)
            .clone()
            .insert_column(SubstModel::N, mu)
            .insert_row(SubstModel::N, 0.0);
        q.fill_diagonal(0.0);
        for i in 0..(SubstModel::N + 1) {
            q[(i, i)] = -q.row(i).sum();
        }
        let pi = SubstitutionModel::get_stationary_distribution(subst_model)
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

    fn get_p(&self, time: f64) -> SubstMatrix {
        (self.q.clone() * time).exp()
    }

    fn get_q(&self) -> &SubstMatrix {
        &self.q
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.q[(self.index[i as usize], self.index[j as usize])]
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.params.pi
    }

    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector {
        let mut probs = self
            .get_stationary_distribution()
            .clone()
            .component_mul(char_encoding);
        if probs.sum() == 0.0 {
            probs.fill_row(SubstModel::N, 1.0);
        } else {
            probs.scale_mut(1.0 / probs.sum());
        }
        probs
    }

    fn index(&self) -> &[usize; 255] {
        &self.index
    }

    fn get_params(&self) -> &PIPParams<SubstModel> {
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

impl<SubstModel: SubstitutionModel> EvoModelInfo for PIPModelInfo<SubstModel>
where
    SubstModel: Clone,
    SubstModel::Params: Clone,
    SubstModel::ModelType: Clone,
{
    type Model = PIPModel<SubstModel>;

    fn new(info: &PhyloInfo, model: &Self::Model) -> Result<Self> {
        if info.msa.is_none() {
            bail!("An MSA is required to set up the likelihood computation.");
        }
        let leaf_count = info.tree.leaves.len();
        let internal_count = info.tree.internals.len();
        let node_count = leaf_count + internal_count;
        let msa = info.msa.as_ref().unwrap();
        let msa_length = msa[0].seq().len();
        let mut leaf_seq_info = info.leaf_encoding.clone();
        for leaf_seq in leaf_seq_info.iter_mut() {
            for mut site_info in leaf_seq.column_iter_mut() {
                site_info.component_mul_assign(model.get_stationary_distribution());
                if site_info.sum() == 0.0 {
                    site_info.fill_row(SubstModel::N, 1.0);
                } else {
                    site_info.scale_mut((1.0) / site_info.sum());
                }
            }
        }
        Ok(PIPModelInfo::<SubstModel> {
            tree_length: info.tree.get_all_branch_lengths().iter().sum(),
            ftilde: leaf_seq_info
                .into_iter()
                .chain(vec![
                    DMatrix::<f64>::zeros(SubstModel::N + 1, msa_length);
                    internal_count
                ])
                .collect::<Vec<DMatrix<f64>>>(),
            ins_probs: vec![0.0; node_count],
            surv_probs: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            p: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_ftilde: vec![DMatrix::<f64>::zeros(SubstModel::N + 1, 1); node_count],
            c0_f: vec![0.0; node_count],
            c0_p: vec![0.0; node_count],
            branches: info
                .tree
                .leaves
                .iter()
                .map(|n| n.blen)
                .chain(info.tree.internals.iter().map(|n| n.blen))
                .collect(),
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
pub struct PIPLikelihoodCost<'a, SubstModel: SubstitutionModel> {
    pub(crate) info: PhyloInfo,
    pub(crate) model: &'a PIPModel<SubstModel>,
}

impl<'a, SubstModel: SubstitutionModel> LikelihoodCostFunction for PIPLikelihoodCost<'a, SubstModel>
where
    SubstModel: Clone,
    SubstModel::Params: Clone,
    SubstModel::ModelType: Clone,
{
    type Model = PIPModel<SubstModel>;
    type Info = PIPModelInfo<SubstModel>;

    fn compute_log_likelihood(&self) -> f64 {
        self.compute_log_likelihood().0
    }

    fn get_empirical_frequencies(&self) -> FreqVector {
        todo!()
    }
}

impl<'a, SubstModel: SubstitutionModel> PIPLikelihoodCost<'a, SubstModel>
where
    SubstModel: Clone,
    SubstModel::ModelType: Clone,
    SubstModel::Params: Clone,
{
    pub(crate) fn compute_log_likelihood(&self) -> (f64, PIPModelInfo<SubstModel>) {
        let mut tmp_info = PIPModelInfo::<SubstModel>::new(&self.info, self.model).unwrap();
        (
            self.compute_log_likelihood_with_tmp(&mut tmp_info),
            tmp_info,
        )
    }
    fn compute_log_likelihood_with_tmp(&self, tmp: &mut PIPModelInfo<SubstModel>) -> f64 {
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                Int(idx) => {
                    if self.info.tree.root == *node_idx {
                        self.set_root_values(*idx, self.model, tmp);
                    }
                    self.set_internal_values(*idx, self.model, tmp);
                }
                Leaf(idx) => {
                    self.set_leaf_values(*idx, self.model, tmp);
                }
            };
        }
        let root_idx = self.get_node_id(&self.info.tree.root);
        let msa_length = tmp.ftilde[0].ncols();

        let nu = self.model.params.lambda * (tmp.tree_length + 1.0 / self.model.params.mu);

        let ln_phi = nu.ln() * msa_length as f64 + (tmp.c0_p[root_idx] - 1.0) * nu
            - (log_factorial(msa_length));
        tmp.p[root_idx].map(|x| x.ln()).sum() + ln_phi
    }

    fn set_root_values(
        &self,
        idx: usize,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let idx = idx + self.info.tree.leaves.len();
        self.compute_model(idx, model, tmp);
        if !tmp.valid[idx] {
            let mu = model.params.mu;
            tmp.surv_probs[idx] = 1.0;
            tmp.ins_probs[idx] = Self::insertion_probability(tmp.tree_length, 1.0 / mu, mu);
            self.compute_int_ftilde(idx, model, tmp);
            self.compute_int_ancestors(idx, tmp);
            tmp.anc[idx].fill_column(0, 1.0);
            self.compute_int_p(idx, tmp);
            self.compute_int_c0(idx, model, tmp);
            tmp.valid[idx] = true;
        }
    }

    fn set_internal_values(
        &self,
        idx: usize,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let idx = idx + self.info.tree.leaves.len();
        self.compute_model(idx, model, tmp);
        if !tmp.valid[idx] {
            let mu = model.params.mu;
            let b = tmp.branches[idx];
            tmp.surv_probs[idx] = Self::survival_probability(mu, b);
            tmp.ins_probs[idx] = Self::insertion_probability(tmp.tree_length, b, mu);
            self.compute_int_ftilde(idx, model, tmp);
            self.compute_int_ancestors(idx, tmp);
            self.compute_int_p(idx, tmp);
            self.compute_int_c0(idx, model, tmp);
            tmp.valid[idx] = true;
        }
    }

    fn set_leaf_values(
        &self,
        idx: usize,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        if !tmp.valid[idx] {
            self.compute_model(idx, model, tmp);
            let mu = model.params.mu;
            let b = tmp.branches[idx];
            tmp.surv_probs[idx] = Self::survival_probability(mu, b);
            tmp.ins_probs[idx] = Self::insertion_probability(tmp.tree_length, b, mu);
            for (i, c) in self.info.msa.as_ref().unwrap()[idx]
                .seq()
                .iter()
                .enumerate()
            {
                if *c != b'-' {
                    tmp.anc[idx][(i, 0)] = 1.0;
                }
            }
            tmp.f[idx] =
                Self::ftilde(&tmp.ftilde[idx], model).component_mul(&tmp.anc[idx].column(0));
            tmp.p[idx] = tmp.f[idx].clone() * tmp.surv_probs[idx] * tmp.ins_probs[idx];
            tmp.c0_ftilde[idx][SubstModel::N] = 1.0;
            tmp.c0_f[idx] = 0.0;
            tmp.c0_p[idx] = (1.0 - tmp.surv_probs[idx]) * tmp.ins_probs[idx];
            tmp.valid[idx] = true;
        }
    }

    fn compute_int_p(&self, idx: usize, tmp: &mut PIPModelInfo<SubstModel>) {
        tmp.p[idx] = tmp.f[idx].clone().component_mul(&tmp.anc[idx].column(0))
            * tmp.surv_probs[idx]
            * tmp.ins_probs[idx];
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        let x_p = tmp.p[x_idx].clone();
        let y_p = tmp.p[y_idx].clone();
        tmp.p[idx] +=
            tmp.anc[idx].column(1).component_mul(&x_p) + tmp.anc[idx].column(2).component_mul(&y_p);
    }

    fn get_node_id(&self, node_idx: &NodeIdx) -> usize {
        match node_idx {
            Int(idx) => idx + self.info.tree.leaves.len(),
            Leaf(idx) => *idx,
        }
    }

    fn compute_int_ftilde(
        &self,
        idx: usize,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = Self::ftilde(&tmp.ftilde[idx], model);
    }

    fn compute_model(
        &self,
        idx: usize,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        if !tmp.models_valid[idx] {
            tmp.models[idx] = model.get_p(tmp.branches[idx]);
            tmp.models_valid[idx] = true;
        }
    }

    fn compute_int_ancestors(&self, idx: usize, tmp: &mut PIPModelInfo<SubstModel>) {
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        let x_anc = tmp.anc[x_idx].clone();
        let y_anc = tmp.anc[y_idx].clone();
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

    fn compute_int_c0(
        &self,
        idx: usize,
        model: &PIPModel<SubstModel>,
        tmp: &mut PIPModelInfo<SubstModel>,
    ) {
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        tmp.c0_ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.c0_ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.c0_ftilde[y_idx]));
        tmp.c0_f[idx] = Self::ftilde(&tmp.c0_ftilde[idx], model)[0];
        tmp.c0_p[idx] = (1.0 + tmp.surv_probs[idx] * (tmp.c0_f[idx] - 1.0)) * tmp.ins_probs[idx]
            + tmp.c0_p[x_idx]
            + tmp.c0_p[y_idx];
    }

    fn ftilde(partial_probs: &DMatrix<f64>, model: &PIPModel<SubstModel>) -> DVector<f64> {
        model.params.pi.transpose().mul(partial_probs).transpose()
    }

    fn insertion_probability(tree_length: f64, b: f64, mu: f64) -> f64 {
        b / (tree_length + 1.0 / mu)
    }

    fn survival_probability(mu: f64, b: f64) -> f64 {
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
mod pip_model_tests;
