use crate::sequences::NUCLEOTIDES_STR;
use crate::Rounding;
use crate::{f64_h, Result};
use anyhow::anyhow;
use bio::io::fasta::Record;
use nalgebra::{Const, DMatrix, DimMin, SMatrix, SVector};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::sequences::NUCLEOTIDES_STR;
use crate::tree::NodeIdx;
use crate::{f64_h, Result, Rounding};

pub mod dna_models;
pub mod protein_models;

pub type SubstMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[derive(Clone, Debug, PartialEq)]
pub struct SubstitutionModel<const N: usize> {
    index: [i32; 255],
    q: SubstMatrix,
    pi: FreqVector,
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
    pub substitution_matrix: SubstMatrix,
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
        let (q, pi) = match model_name.to_uppercase().as_str() {
            "JC69" => dna_models::jc69(model_params)?,
            "K80" => dna_models::k80(model_params)?,
            "TN93" => dna_models::tn93(model_params)?,
            "GTR" => dna_models::gtr(model_params)?,
            _ => bail!("Unknown DNA model requested."),
        };
        let mut model = DNASubstModel {
            index: dna_models::nucleotide_index(),
            q,
            pi,
        };
        Ok(model)
    }
}

impl ProteinSubstModel {
    pub fn new(model_name: &str, model_params: &[f64]) -> Result<Self> {
        let (q, pi) = match model_name.to_uppercase().as_str() {
            "WAG" => protein_models::wag()?,
            "BLOSUM" => protein_models::blosum()?,
            "HIVB" => protein_models::hivb()?,
            _ => bail!("Unknown protein model requested."),
        };
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
    fn get_p(&self, time: f64) -> SubstMatrix {
        (self.q.clone() * time).exp()
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        HashMap::<f64_h, (SubstMatrix, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.get_scoring_matrix_corrected(time, zero_diag, rounding),
            )
        }))
    }

    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix_corrected(time, false, rounding)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.pi
    }

    fn get_scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64) {
        let p = self.get_p(time);
        let mut scores = p.map(|x| -x.ln());
        if rounding.round {
            scores = scores.map(|x| {
                (x * 10.0_f64.powf(rounding.digits as f64)).round()
                    / 10.0_f64.powf(rounding.digits as f64)
            });
        }
        if zero_diag {
            scores.fill_diagonal(0.0);
        }

        (scores, scores.mean())
    }
}

#[cfg(test)]
mod substitution_models_tests;
