use std::collections::HashMap;

use anyhow::bail;
use nalgebra::dvector;
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvolutionaryModel;
use crate::substitution_models::dna_models::nucleotide_index;
use crate::substitution_models::FreqVector;
use crate::substitution_models::{dna_models::DNASubstModel, SubstMatrix, SubstitutionModel};
use crate::{Result, Rounding};

#[derive(Clone, Debug)]
pub struct PIPModel<const N: usize> {
    pub index: [i32; 255],
    pub subst_model: SubstitutionModel<N>,
    pub lambda: f64,
    pub mu: f64,
    pub q: SubstMatrix,
    pub pi: FreqVector,
}

// TODO: Make sure Q matrix makes sense like this ALL the time.
impl EvolutionaryModel<4> for PIPModel<4> {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        if model_params.len() < 2 {
            bail!("Too few values provided for PIP, required 2 values, lambda and mu.");
        }
        let lambda = model_params[0];
        let mu = model_params[1];
        let subst_model = DNASubstModel::new(model_name, &model_params[2..])?;
        let mut index = nucleotide_index();
        index[b'-' as usize] = 4;
        let mut q = subst_model
            .q
            .clone()
            .insert_column(4, mu)
            .insert_row(4, 0.0);
        q.fill_diagonal(0.0);
        for i in 0..5 {
            q[(i, i)] = -q.row(i).sum();
        }
        let pi = subst_model.pi.clone().insert_row(4, 0.0);
        let model = PIPModel {
            index,
            subst_model,
            lambda,
            mu,
            q,
            pi,
        };
        Ok(model)
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

    fn generate_scorings(
        &self,
        _: &[f64],
        _: bool,
        _: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        unreachable!("This should not be called.")
    }

    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn get_scoring_matrix(&self, _: f64, _: &Rounding) -> (SubstMatrix, f64) {
        unreachable!("This should not be called.")
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.pi
    }

    fn get_char_probability(&self, char: u8) -> FreqVector {
        let freq = if char == b'-' {
            dvector![0.0, 0.0, 0.0, 0.0, 1.0]
        } else {
            self.subst_model
                .get_char_probability(char)
                .insert_row(4, 0.0)
        };
        debug_assert_eq!(freq.len(), 5);
        freq
    }
}

#[cfg(test)]
mod pip_model_tests;
