use crate::sequences::NUCLEOTIDES_STR;
use crate::{f64_h, Result};
use anyhow::anyhow;
use bio::io::fasta::Record;
use nalgebra::{Const, DMatrix, DimMin, SMatrix, SVector};
use ordered_float::OrderedFloat;
use std::collections::HashMap;

pub mod dna_models;
pub mod protein_models;

pub type SubstMatrix<const N: usize> = SMatrix<f64, N, N>;
pub type FreqVector<const N: usize> = SVector<f64, N>;

#[derive(Clone, Debug, PartialEq)]
pub struct SubstitutionModel<const N: usize> {
    index: [i32; 255],
    pub q: SubstMatrix<N>,
    pub pi: FreqVector<N>,
}

pub type DNASubstModel = SubstitutionModel<4>;
pub type ProteinSubstModel = SubstitutionModel<20>;

pub trait EvolutionaryModelNodeInfo<const N: usize> {
    fn get_leaf_info(sequence: Record, branch_length: f64, model: &SubstitutionModel<N>) -> Self;
    fn get_internal_info(
        childx: &Self,
        childy: &Self,
        branch_length: f64,
        model: &SubstitutionModel<N>,
    ) -> Self;
}

pub struct DNAModelNodeInfo {
    pub partial_likelihoods: DMatrix<f64>,
    pub partial_likelihoods_valid: bool,
    pub substitution_matrix: SubstMatrix<4>,
}

impl DNAModelNodeInfo {
    fn new(sites: usize, branch_length: f64, model: &DNASubstModel) -> Self {
        Self {
            partial_likelihoods: DMatrix::zeros(4, sites),
            partial_likelihoods_valid: false,
            substitution_matrix: model.get_p(branch_length),
        }
    }
}

impl EvolutionaryModelNodeInfo<4> for DNAModelNodeInfo {
    fn get_leaf_info(record: Record, branch_length: f64, model: &DNASubstModel) -> Self {
        let sites = record.seq().len();
        let mut info = Self::new(sites, branch_length, model);
        let char_probabilities = DMatrix::from_fn(4, sites, |i, j| match record.seq()[j] {
            b'-' => model.pi[i],
            _ => {
                if NUCLEOTIDES_STR.find(record.seq()[j] as char).unwrap() == i {
                    1.0
                } else {
                    0.0
                }
            }
        });
        char_probabilities.mul_to(&info.substitution_matrix, &mut info.partial_likelihoods);
        info.partial_likelihoods_valid = true;
        info
    }

    fn get_internal_info(
        childx: &Self,
        childy: &Self,
        branch_length: f64,
        model: &DNASubstModel,
    ) -> Self {
        let char_probabilities = childx
            .partial_likelihoods
            .component_mul(&childy.partial_likelihoods);
        let mut info = Self::new(char_probabilities.ncols(), branch_length, model);
        char_probabilities.mul_to(&info.substitution_matrix, &mut info.partial_likelihoods);
        info.partial_likelihoods_valid = true;
        info
    }
}

impl DNASubstModel {
    pub fn new(model_name: &str, model_params: &[f64]) -> Result<Self> {
        let q: SubstMatrix<4>;
        let pi: FreqVector<4>;
        match model_name.to_uppercase().as_str() {
            "JC69" => (q, pi) = dna_models::jc69(model_params)?,
            "K80" => (q, pi) = dna_models::k80(model_params)?,
            "TN93" => (q, pi) = dna_models::tn93(model_params)?,
            "GTR" => (q, pi) = dna_models::gtr(model_params)?,
            _ => return Err(anyhow!("Unknown DNA model requested.")),
        }
        let model = DNASubstModel {
            index: dna_models::nucleotide_index(),
            q,
            pi,
        };
        Ok(model)
    }
}

impl ProteinSubstModel {
    pub fn new(model_name: &str) -> Result<Self> {
        let q: SubstMatrix<20>;
        let pi: FreqVector<20>;
        match model_name.to_uppercase().as_str() {
            "WAG" => (q, pi) = protein_models::wag()?,
            "BLOSUM" => (q, pi) = protein_models::blosum()?,
            "HIVB" => (q, pi) = protein_models::hivb()?,
            _ => return Err(anyhow!("Unknown protein model requested.")),
        }
        Ok(ProteinSubstModel {
            index: protein_models::aminoacid_index(),
            q,
            pi,
        })
    }
}

impl<const N: usize> SubstitutionModel<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    pub fn get_p(&self, time: f64) -> SubstMatrix<N> {
        (self.q * time).exp()
    }

    pub fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    pub fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix<N>, f64)> {
        HashMap::<f64_h, (SubstMatrix<N>, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.get_scoring_matrix_corrected(time, zero_diag, rounded),
            )
        }))
    }

    pub fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    pub fn get_scoring_matrix(&self, time: f64, rounded: bool) -> (SubstMatrix<N>, f64) {
        self.get_scoring_matrix_corrected(time, false, rounded)
    }

    fn get_scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounded: bool,
    ) -> (SubstMatrix<N>, f64) {
        let p = self.get_p(time);
        let mapping = if rounded {
            |x: f64| (-x.ln().round())
        } else {
            |x: f64| -x.ln()
        };
        let mut scores = p.map(mapping);
        if zero_diag {
            scores.fill_diagonal(0.0);
        }
        (scores, scores.mean())
    }
}

#[cfg(test)]
mod substitution_models_tests;
