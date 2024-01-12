use std::collections::HashMap;

use anyhow::bail;
use lazy_static::lazy_static;
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvolutionaryModel;
use crate::likelihood::LikelihoodCostFunction;
use crate::sequences::{charify, dna_alphabet, NUCLEOTIDES_STR};
use crate::substitution_models::{
    FreqVector, ParsimonyModel, SubstMatrix, SubstParams, SubstitutionLikelihoodCost,
    SubstitutionModel, SubstitutionModelInfo,
};
use crate::{Result, Rounding};

pub type DNASubstModel = SubstitutionModel<4>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, 4>;
pub type DNASubstModelInfo = SubstitutionModelInfo<4>;

mod dna_substitution_parameters;
pub(crate) use dna_substitution_parameters::*;
pub mod dna_model_optimiser;

mod gtr;
mod hky;
mod jc69;
mod k80;
mod tn93;
pub use gtr::*;
pub use hky::*;
pub use jc69::*;
pub use k80::*;
pub use tn93::*;

lazy_static! {
    pub static ref DNA_SETS: Vec<FreqVector> = {
        let mut map = Vec::<FreqVector>::new();
        map.resize(255, FreqVector::from_element(4, 1.0 / 4.0));
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8 as char;
            elem.set_column(
                0,
                &match char {
                    'T' | 't' => FreqVector::from_column_slice(&[1.0, 0.0, 0.0, 0.0]),
                    'C' | 'c' => FreqVector::from_column_slice(&[0.0, 1.0, 0.0, 0.0]),
                    'A' | 'a' => FreqVector::from_column_slice(&[0.0, 0.0, 1.0, 0.0]),
                    'G' | 'g' => FreqVector::from_column_slice(&[0.0, 0.0, 0.0, 1.0]),
                    'V' | 'v' => {
                        FreqVector::from_column_slice(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
                    }
                    'D' | 'd' => {
                        FreqVector::from_column_slice(&[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0])
                    }
                    'B' | 'b' => {
                        FreqVector::from_column_slice(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
                    }
                    'H' | 'h' => {
                        FreqVector::from_column_slice(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
                    }
                    'M' | 'm' => FreqVector::from_column_slice(&[0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0]),
                    'R' | 'r' => FreqVector::from_column_slice(&[0.0, 0.0, 1.0 / 2.0, 1.0 / 2.0]),
                    'W' | 'w' => FreqVector::from_column_slice(&[1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0]),
                    'S' | 's' => FreqVector::from_column_slice(&[0.0, 1.0 / 2.0, 0.0, 1.0 / 2.0]),
                    'Y' | 'y' => FreqVector::from_column_slice(&[1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0]),
                    'K' | 'k' => FreqVector::from_column_slice(&[1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0]),
                    _ => continue,
                },
            );
        }
        map
    };
}

lazy_static! {
    pub static ref NUCLEOTIDE_INDEX: [i32; 255] = {
        let mut index = [-1_i32; 255];
        for (i, char) in charify(NUCLEOTIDES_STR).into_iter().enumerate() {
            index[char as usize] = i as i32;
            index[char.to_ascii_lowercase() as usize] = i as i32;
        }
        index
    };
}

fn make_dna_model(params: DNASubstParams, q: SubstMatrix) -> DNASubstModel {
    let pi = params.pi.clone();
    DNASubstModel {
        params: SubstParams::DNA(params),
        index: *NUCLEOTIDE_INDEX,
        q,
        pi,
    }
}

impl EvolutionaryModel<4> for DNASubstModel {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        match model_name.to_uppercase().as_str() {
            "JC69" => Ok(jc69(parse_jc69_parameters(model_params)?)),
            "K80" => Ok(k80(parse_k80_parameters(model_params)?)),
            "HKY" => Ok(hky(parse_hky_parameters(model_params)?)),
            "TN93" => Ok(tn93(parse_tn93_parameters(model_params)?)),
            "GTR" => Ok(gtr(parse_gtr_parameters(model_params)?)),
            _ => bail!("Unknown DNA model requested."),
        }
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
        let mut probs = FreqVector::zeros(4);
        if NUCLEOTIDES_STR.contains(char as char) {
            probs[self.index[char as usize] as usize] = 1.0;
        } else {
            probs = FreqVector::from_column_slice(self.get_stationary_distribution().as_slice())
                .component_mul(&DNA_SETS[char as usize]);
        }
        probs.scale_mut(1.0 / probs.sum());
        probs
    }
}

impl ParsimonyModel<4> for DNASubstModel {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        self.generate_scorings(times, zero_diag, rounding)
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix(time, rounding)
    }
}

impl<'a> LikelihoodCostFunction<'a, 4> for DNALikelihoodCost<'a> {
    type Model = DNASubstModel;
    type Info = SubstitutionModelInfo<4>;

    fn compute_log_likelihood(&self, model: &Self::Model) -> f64 {
        self.compute_log_likelihood(model).0
    }

    fn get_empirical_frequencies(&self) -> FreqVector {
        let all_counts = self.info.get_counts(&dna_alphabet());
        let mut total = all_counts.values().sum::<f64>();
        let index = &NUCLEOTIDE_INDEX;
        let mut freqs = FreqVector::zeros(4);
        for (&char, &count) in all_counts.iter() {
            if index[char as usize] >= 0 {
                freqs[index[char as usize] as usize] += count;
            } else {
                freqs += &DNA_SETS[char as usize].scale(count);
            }
        }
        for &char in NUCLEOTIDES_STR.as_bytes() {
            if freqs[index[char as usize] as usize] == 0.0 {
                freqs[index[char as usize] as usize] += 1.0;
                total += 1.0;
            }
        }
        freqs.map(|x| x / total)
    }
}

fn make_pi(pi_array: &[f64]) -> Result<FreqVector> {
    let pi = FreqVector::from_column_slice(pi_array);
    debug_assert!(
        pi.len() == 4,
        "There have to be 4 equilibrium frequencies for DNA models."
    );
    if pi.sum() != 1.0 {
        bail!("The equilibrium frequencies provided do not sum up to 1.");
    }
    Ok(pi)
}

#[cfg(test)]
mod dna_optimisation_tests;
