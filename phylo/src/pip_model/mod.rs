use std::ops::Mul;
use std::vec;

use anyhow::bail;
use nalgebra::{Const, DMatrix, DVector, DimMin};

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::dna_models::NUCLEOTIDE_INDEX;
use crate::substitution_models::protein_models::{ProteinSubstModel, AMINOACID_INDEX};
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
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let (lambda, mu) = PIPModel::<4>::check_pip_params(model_params)?;
        let subst_model = DNASubstModel::new(model_name, &model_params[2..])?;
        let index = nucleotide_index();
        Ok(PIPModel::make_pip(index, subst_model, mu, lambda))
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

    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector {
        let mut probs = self
            .get_stationary_distribution()
            .clone()
            .component_mul(char_encoding);
        if probs.sum() == 0.0 {
            probs.fill_row(4, 1.0);
        } else {
            probs.scale_mut(1.0 / probs.sum());
        }
        probs
    }
}

impl EvolutionaryModel<20> for PIPModel<20> {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let (lambda, mu) = PIPModel::<20>::check_pip_params(model_params)?;
        let subst_model = ProteinSubstModel::new(model_name, &model_params[2..])?;
        let index = aminoacid_index();
        Ok(PIPModel::make_pip(index, subst_model, mu, lambda))
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

    fn get_char_probability(&self, char_encoding: &FreqVector) -> FreqVector {
        let mut probs = self
            .get_stationary_distribution()
            .clone()
            .component_mul(char_encoding);
        if probs.sum() == 0.0 {
            probs.fill_row(20, 1.0);
        } else {
            probs.scale_mut(1.0 / probs.sum());
        }
        probs
    }
}

#[derive(Debug)]
pub(crate) struct PIPLikelihoodCost<'a, const N: usize> {
    pub(crate) info: &'a PhyloInfo,
}

#[derive(Debug)]
pub(crate) struct PIPModelInfo<const N: usize> {
    surv_ins_weights: Vec<f64>,
    anc: Vec<DMatrix<f64>>,
    branches: Vec<f64>,
    ftilde: Vec<DMatrix<f64>>,
    f: Vec<DVector<f64>>,
    pnu: Vec<DVector<f64>>,
    c0_f1: Vec<f64>,
    c0_pnu: Vec<f64>,
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
        let mut leaf_seq_info = info.leaf_encoding.clone();
        for leaf_seq in leaf_seq_info.iter_mut() {
            for mut site_info in leaf_seq.column_iter_mut() {
                site_info.component_mul_assign(model.get_stationary_distribution());
                if site_info.sum() == 0.0 {
                    site_info.fill_row(N, 1.0);
                } else {
                    site_info.scale_mut((1.0) / site_info.sum());
                }
            }
        }
        Ok(PIPModelInfo {
            ftilde: leaf_seq_info
                .into_iter()
                .chain(vec![
                    DMatrix::<f64>::zeros(N + 1, msa_length);
                    internal_count
                ])
                .collect::<Vec<DMatrix<f64>>>(),
            surv_ins_weights: vec![0.0; node_count],
            f: vec![DVector::<f64>::zeros(msa_length); node_count],
            pnu: vec![DVector::<f64>::zeros(msa_length); node_count],
            c0_f1: vec![0.0; node_count],
            c0_pnu: vec![0.0; node_count],
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

impl<'a> LikelihoodCostFunction<'a, 4> for PIPLikelihoodCost<'a, 4> {
    type Model = PIPModel<4>;
    type Info = PIPModelInfo<4>;

    fn compute_log_likelihood(&self, model: &Self::Model) -> f64 {
        let mut tmp_info = PIPModelInfo::new(self.info, model).unwrap();
        self.compute_log_likelihood_with_tmp(model, &mut tmp_info)
    }

    fn get_empirical_frequencies(&self) -> FreqVector {
        todo!()
    }
}

impl<'a> LikelihoodCostFunction<'a, 20> for PIPLikelihoodCost<'a, 20> {
    type Model = PIPModel<20>;
    type Info = PIPModelInfo<20>;

    fn compute_log_likelihood(&self, model: &Self::Model) -> f64 {
        let mut tmp_info = PIPModelInfo::new(self.info, model).unwrap();
        self.compute_log_likelihood_with_tmp(model, &mut tmp_info)
    }
}

fn log_factorial(n: usize) -> f64 {
    (1..n + 1).map(|i| (i as f64).ln()).sum::<f64>()
}

impl<const N: usize> PIPLikelihoodCost<'_, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn compute_log_likelihood_with_tmp(
        &self,
        model: &PIPModel<N>,
        tmp: &mut PIPModelInfo<N>,
    ) -> f64 {
        for node_idx in &self.info.tree.postorder {
            match node_idx {
                Int(idx) => {
                    if self.info.tree.root == *node_idx {
                        self.set_root_values(*idx, model, tmp);
                        self.set_root_values(*idx, model, tmp);
                    }
                    self.set_internal_values(*idx, model, tmp);
                    self.set_internal_values(*idx, model, tmp);
                }
                Leaf(idx) => {
                    self.set_leaf_values(*idx, model, tmp);
                    self.set_leaf_values(*idx, model, tmp);
                }
            };
        }
        let root_idx = self.get_node_id(&self.info.tree.root);
        let msa_length = self.tmp.ftilde[0].ncols();
        self.tmp.pnu[root_idx].map(|x| x.ln()).sum() + self.tmp.c0_pnu[root_idx]
            - log_factorial_shifted(msa_length)
    }
}

fn log_factorial_shifted(n: usize) -> f64 {
    // An approximation using Stirling's formula, minus constant log(sqrt(2*PI)).
    // Has an absolute error of 3e-4 for n=2, 3e-6 for n=10, 3e-9 for n=100.
    let n = n as f64;
    n * n.ln() - n + n.ln() / 2.0 + 1.0 / (12.0 * n)
}

impl<'a, const N: usize> PIPLikelihoodCost<'a, N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn set_root_values(&mut self, idx: usize) {
        let idx = idx + self.info.tree.leaves.len();
        self.compute_model(idx);
        if !self.tmp.valid[idx] {
            self.tmp.surv_ins_weights[idx] = self.model.lambda / self.model.mu;
            self.compute_int_ftilde(idx);
            self.compute_int_ancestors(idx);
            self.tmp.anc[idx].fill_column(0, 1.0);
            self.compute_int_pnu(idx);
            self.compute_int_c0(idx);
            self.tmp.valid[idx] = true;
        }
    }

    fn set_internal_values(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
    fn set_internal_values(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        let idx = idx + self.info.tree.leaves.len();
        self.compute_model(idx);
        if !self.tmp.valid[idx] {
            self.tmp.surv_ins_weights[idx] = Self::survival_insertion_weight(
                self.model.lambda,
                self.model.mu,
                self.tmp.branches[idx],
            );
            self.compute_int_ftilde(idx);
            self.compute_int_ancestors(idx);
            self.compute_int_pnu(idx);
            self.compute_int_c0(idx);
            self.tmp.valid[idx] = true;
        }
    }

    fn set_leaf_values(&mut self, idx: usize) {
        if !self.tmp.valid[idx] {
            self.compute_model(idx);
            self.tmp.surv_ins_weights[idx] = Self::survival_insertion_weight(
                self.model.lambda,
                self.model.mu,
                self.tmp.branches[idx],
            );
            for (i, c) in self.info.msa.as_ref().unwrap()[idx]
                .seq()
                .iter()
                .enumerate()
            {
                if *c != b'-' {
                    tmp.anc[idx][(i, 0)] = 1.0;
                    tmp.anc[idx][(i, 0)] = 1.0;
                }
            }
            self.tmp.f[idx] = self
                .ftilde(&self.tmp.ftilde[idx])
                .component_mul(&self.tmp.anc[idx].column(0));
            self.tmp.pnu[idx] = self.tmp.f[idx].clone() * self.tmp.surv_ins_weights[idx];
            self.tmp.c0_f1[idx] = -1.0;
            self.tmp.c0_pnu[idx] = -self.tmp.surv_ins_weights[idx];
            self.tmp.valid[idx] = true;
        }
    }

    fn compute_int_pnu(&mut self, idx: usize) {
        self.tmp.pnu[idx] = self.tmp.f[idx]
            .clone()
            .component_mul(&self.tmp.anc[idx].column(0))
            * self.tmp.surv_ins_weights[idx];
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        let x_pnu = self.tmp.pnu[x_idx].clone();
        let y_pnu = self.tmp.pnu[y_idx].clone();
        self.tmp.pnu[idx] += self.tmp.anc[idx].column(1).component_mul(&x_pnu)
            + self.tmp.anc[idx].column(2).component_mul(&y_pnu);
    }

    fn get_node_id(&self, node_idx: &NodeIdx) -> usize {
        match node_idx {
            Int(idx) => idx + self.info.tree.leaves.len(),
            Leaf(idx) => *idx,
        }
    }

    fn compute_int_ftilde(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
    fn compute_int_ftilde(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = Self::ftilde(&tmp.ftilde[idx], model);
        tmp.ftilde[idx] = (&tmp.models[x_idx])
            .mul(&tmp.ftilde[x_idx])
            .component_mul(&(&tmp.models[y_idx]).mul(&tmp.ftilde[y_idx]));
        tmp.f[idx] = Self::ftilde(&tmp.ftilde[idx], model);
    }

    fn compute_model(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        if !tmp.models_valid[idx] {
            tmp.models[idx] = model.get_p(tmp.branches[idx]);
            tmp.models_valid[idx] = true;
    fn compute_model(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        if !tmp.models_valid[idx] {
            tmp.models[idx] = model.get_p(tmp.branches[idx]);
            tmp.models_valid[idx] = true;
        }
    }

    fn compute_int_ancestors(&self, idx: usize, tmp: &mut PIPModelInfo<N>) {
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
    fn compute_int_c0(&self, idx: usize, model: &PIPModel<N>, tmp: &mut PIPModelInfo<N>) {
        let node = &self.info.tree.internals[idx - self.info.tree.leaves.len()];
        let x_idx = self.get_node_id(&node.children[0]);
        let y_idx = self.get_node_id(&node.children[1]);
        let mu = self.model.mu;
        self.tmp.c0_f1[idx] = (1.0
            + (-mu * self.tmp.branches[x_idx]).exp() * self.tmp.c0_f1[x_idx])
            * (1.0 + (-mu * self.tmp.branches[y_idx]).exp() * self.tmp.c0_f1[y_idx])
            - 1.0;
        self.tmp.c0_pnu[idx] = self.tmp.surv_ins_weights[idx] * self.tmp.c0_f1[idx]
            + self.tmp.c0_pnu[x_idx]
            + self.tmp.c0_pnu[y_idx];
    }

    fn ftilde(partial_probs: &DMatrix<f64>, model: &PIPModel<N>) -> DVector<f64> {
        model.pi.transpose().mul(partial_probs).transpose()
    fn ftilde(partial_probs: &DMatrix<f64>, model: &PIPModel<N>) -> DVector<f64> {
        model.pi.transpose().mul(partial_probs).transpose()
    }

    fn survival_insertion_weight(lambda: f64, mu: f64, b: f64) -> f64 {
        //A function equal to old
        //nu * insertion_probability(tree_length, b, mu) * survival_probablitily(mu, b)
        lambda / mu * (1.0 - (-b * mu).exp())
    }
}

#[cfg(test)]
mod pip_model_tests;
