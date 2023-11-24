use std::ops::Mul;
use std::vec;

use anyhow::bail;
use nalgebra::{Const, DMatrix, DVector, DimMin};

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::dna_models::nucleotide_index;
use crate::substitution_models::protein_models::{aminoacid_index, ProteinSubstModel};
use crate::substitution_models::FreqVector;
use crate::substitution_models::{dna_models::DNASubstModel, SubstMatrix, SubstitutionModel};
use crate::tree::NodeIdx::{self, Internal as Int, Leaf};
use crate::Result;

#[derive(Clone, Debug)]
pub struct PIPModel<const N: usize> {
    pub index: [i32; 255],
    pub subst_model: SubstitutionModel<N>,
    pub lambda: f64,
    pub mu: f64,
    pub q: SubstMatrix,
    pub pi: FreqVector,
}

impl<const N: usize> PIPModel<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        debug_assert!(time >= 0.0);
        (self.q.clone() * time).exp()
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.pi
    }

    fn make_pip(
        index: [i32; 255],
        subst_model: SubstitutionModel<N>,
        mu: f64,
        lambda: f64,
    ) -> PIPModel<N> {
        let mut index = index;
        index[b'-' as usize] = N as i32;
        let mut q = subst_model
            .q
            .clone()
            .insert_column(N, mu)
            .insert_row(N, 0.0);
        q.fill_diagonal(0.0);
        for i in 0..(N + 1) {
            q[(i, i)] = -q.row(i).sum();
        }
        let pi = subst_model.pi.clone().insert_row(N, 0.0);
        PIPModel {
            index,
            subst_model,
            lambda,
            mu,
            q,
            pi,
        }
    }

    fn check_pip_params(model_params: &[f64]) -> Result<(f64, f64)> {
        if model_params.len() < 2 {
            bail!("Too few values provided for PIP, required 2 values, lambda and mu.");
        }
        let lambda = model_params[0];
        let mu = model_params[1];
        Ok((lambda, mu))
    }
}

// TODO: Make sure Q matrix makes sense like this ALL the time.
impl EvolutionaryModel<4> for PIPModel<4> {
    fn new(model_name: &str, model_params: &[f64], normalise: bool) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let (lambda, mu) = PIPModel::<4>::check_pip_params(model_params)?;
        let subst_model = DNASubstModel::new(model_name, &model_params[2..], normalise)?;
        let index = nucleotide_index();
        Ok(PIPModel::make_pip(index, subst_model, mu, lambda))
    }

    fn normalise(&mut self) {
        self.normalise();
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        self.get_p(time)
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.get_rate(i, j)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        self.get_stationary_distribution()
    }

    fn get_char_probability(&self, char: u8) -> FreqVector {
        if char == b'-' {
            let mut probs = FreqVector::from_column_slice(&[0.0; 5]);
            probs[4] = 1.0;
            probs
        } else {
            self.subst_model
                .get_char_probability(char)
                .insert_row(4, 0.0)
        }
    }
}

impl EvolutionaryModel<20> for PIPModel<20> {
    fn new(model_name: &str, model_params: &[f64], normalise: bool) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let (lambda, mu) = PIPModel::<20>::check_pip_params(model_params)?;
        let subst_model = ProteinSubstModel::new(model_name, &model_params[2..], normalise)?;
        let index = aminoacid_index();
        Ok(PIPModel::make_pip(index, subst_model, mu, lambda))
    }

    fn normalise(&mut self) {
        self.normalise();
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        self.get_p(time)
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.get_rate(i, j)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        self.get_stationary_distribution()
    }

    fn get_char_probability(&self, char: u8) -> FreqVector {
        if char == b'-' {
            let mut probs = FreqVector::from_column_slice(&[0.0; 21]);
            probs[20] = 1.0;
            probs
        } else {
            self.subst_model
                .get_char_probability(char)
                .insert_row(20, 0.0)
        }
    }
}

#[derive(Debug)]
pub(crate) struct PIPLikelihoodCost<'a, const N: usize> {
    pub(crate) info: &'a PhyloInfo,
}

#[derive(Debug)]
pub(crate) struct PIPModelInfo<const N: usize> {
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

impl<const N: usize> EvolutionaryModelInfo<N> for PIPModelInfo<N> {
    fn new(info: &PhyloInfo, model: &dyn EvolutionaryModel<N>) -> Result<Self> {
        if info.msa.is_none() {
            bail!("An MSA is required to set up the likelihood computation.");
        }
        let leaf_count = info.tree.leaves.len();
        let internal_count = info.tree.internals.len();
        let node_count = leaf_count + internal_count;
        let msa = info.msa.as_ref().unwrap();
        let msa_length = msa[0].seq().len();
        let leaf_seq_info = msa
            .iter()
            .map(|rec| {
                DMatrix::from_columns(
                    rec.seq()
                        .iter()
                        .map(|&c| model.get_char_probability(c))
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
            })
            .collect::<Vec<_>>();
        Ok(PIPModelInfo {
            tree_length: info.tree.get_all_branch_lengths().iter().sum(),
            ftilde: leaf_seq_info
                .into_iter()
                .chain(vec![
                    DMatrix::<f64>::zeros(N + 1, msa_length);
                    internal_count
                ])
                .collect::<Vec<DMatrix<f64>>>(),
            ins_probs: vec![0.0; node_count],
            surv_probs: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            p: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_ftilde: vec![DMatrix::<f64>::zeros(N + 1, 1); node_count],
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
            models: vec![SubstMatrix::zeros(N + 1, N + 1); node_count],
            models_valid: vec![false; node_count],
        })
    }
}

impl<'a> LikelihoodCostFunction<'a, 4> for PIPLikelihoodCost<'a, 4> {
    type Model = PIPModel<4>;
    type Info = PIPModelInfo<4>;

    fn compute_log_likelihood(&self, model: &Self::Model, tmp_info: &mut Self::Info) -> f64 {
        self.compute_log_likelihood(model, tmp_info)
    }
}

fn log_factorial(n: usize) -> f64 {
    (1..n + 1).map(|i| (i as f64).ln()).sum::<f64>()
}

impl<const N: usize> PIPLikelihoodCost<'_, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn compute_log_likelihood(&self, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) -> f64 {
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                Int(idx) => {
                    if self.info.tree.root == *node_idx {
                        self.set_root_values(*idx, model, tmp);
                    }
                    self.set_internal_values(*idx, model, tmp);
                }
                Leaf(idx) => {
                    self.set_leaf_values(*idx, model, tmp);
                }
            };
        }
        let root_idx = self.get_node_id(&self.info.tree.root);
        let msa_length = tmp.ftilde[0].ncols();
        let nu = model.lambda * (tmp.tree_length + 1.0 / model.mu);
        let ln_phi = nu.ln() * msa_length as f64 + (tmp.c0_p[root_idx] - 1.0) * nu
            - (log_factorial(msa_length));
        tmp.p[root_idx].map(|x| x.ln()).sum() + ln_phi
    }

    fn set_root_values(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        let idx = idx + self.info.tree.leaves.len();
        self.compute_model(idx, model, tmp);
        if !tmp.valid[idx] {
            let mu = model.mu;
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

    fn set_internal_values(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        let idx = idx + self.info.tree.leaves.len();
        self.compute_model(idx, model, tmp);
        if !tmp.valid[idx] {
            let mu = model.mu;
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

    fn set_leaf_values(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        if !tmp.valid[idx] {
            self.compute_model(idx, model, tmp);
            let mu = model.mu;
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
            tmp.c0_ftilde[idx][N] = 1.0;
            tmp.c0_f[idx] = 0.0;
            tmp.c0_p[idx] = (1.0 - tmp.surv_probs[idx]) * tmp.ins_probs[idx];
            tmp.valid[idx] = true;
        }
    }

    fn compute_int_p(&self, idx: usize, tmp: &mut PIPModelInfo<N>) {
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

    fn compute_int_ftilde(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = Self::ftilde(&tmp.ftilde[idx], model);
    }

    fn compute_model(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        if !tmp.models_valid[idx] {
            tmp.models[idx] = model.get_p(tmp.branches[idx]);
            tmp.models_valid[idx] = true;
        }
    }

    fn compute_int_ancestors(&self, idx: usize, tmp: &mut PIPModelInfo<N>) {
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

    fn compute_int_c0(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
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

    fn ftilde(partial_probs: &DMatrix<f64>, model: &PIPModel<N>) -> DVector<f64> {
        model.pi.transpose().mul(partial_probs).transpose()
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

#[cfg(test)]
mod pip_model_tests;
