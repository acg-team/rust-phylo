use std::collections::HashMap;

use anyhow::bail;
use log::warn;
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::sequences::{charify, dna_alphabet, AMBIG, NUCLEOTIDES_STR};
use crate::substitution_models::{
    FreqVector, ParsimonyModel, SubstMatrix, SubstitutionLikelihoodCost, SubstitutionModel,
    SubstitutionModelInfo,
};
use crate::{Result, Rounding};

pub type DNASubstModel = SubstitutionModel<4>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, 4>;
pub type DNASubstModelInfo = SubstitutionModelInfo<4>;

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

fn make_dna_model(params: Vec<f64>, q: SubstMatrix, pi: FreqVector) -> DNASubstModel {
    DNASubstModel {
        params,
        index: nucleotide_index(),
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
            "JC69" => jc69(model_params),
            "K80" => k80(model_params),
            "HKY" => hky(model_params),
            "TN93" => tn93(model_params),
            "GTR" => gtr(model_params),
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
        let mut vec = FreqVector::zeros(4);
        if NUCLEOTIDES_STR.contains(char as char) {
            vec[self.index[char as usize] as usize] = 1.0;
        } else {
            vec = FreqVector::from_column_slice(self.get_stationary_distribution().as_slice());
            match char {
                b'V' => vec[self.index[b'T' as usize] as usize] = 0.0,
                b'D' => vec[self.index[b'C' as usize] as usize] = 0.0,
                b'B' => vec[self.index[b'A' as usize] as usize] = 0.0,
                b'H' => vec[self.index[b'G' as usize] as usize] = 0.0,
                b'M' => {
                    vec[self.index[b'T' as usize] as usize] = 0.0;
                    vec[self.index[b'G' as usize] as usize] = 0.0
                }
                b'R' => {
                    vec[self.index[b'T' as usize] as usize] = 0.0;
                    vec[self.index[b'C' as usize] as usize] = 0.0
                }
                b'W' => {
                    vec[self.index[b'G' as usize] as usize] = 0.0;
                    vec[self.index[b'C' as usize] as usize] = 0.0
                }
                b'S' => {
                    vec[self.index[b'T' as usize] as usize] = 0.0;
                    vec[self.index[b'A' as usize] as usize] = 0.0
                }
                b'Y' => {
                    vec[self.index[b'A' as usize] as usize] = 0.0;
                    vec[self.index[b'G' as usize] as usize] = 0.0
                }
                b'K' => {
                    vec[self.index[b'C' as usize] as usize] = 0.0;
                    vec[self.index[b'A' as usize] as usize] = 0.0
                }
                _ => {
                    warn!("Unknown character {} encountered, treating it as X.", char);
                    vec =
                        FreqVector::from_column_slice(self.get_stationary_distribution().as_slice())
                }
            };
        }
        vec.scale_mut(1.0 / vec.sum());
        vec
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

    fn compute_log_likelihood(&self, model: &Self::Model, tmp_info: &mut Self::Info) -> f64 {
        let logl = self.compute_log_likelihood(model, tmp_info);
        tmp_info.reset();
        logl
    }

    fn get_empirical_frequencies(&self) -> FreqVector {
        let all_counts = self.info.get_empirical_frequencies(&dna_alphabet());
        let index = nucleotide_index();
        let dna_ambiguous_chars = dna_ambiguous_chars();
        let mut freqs = FreqVector::zeros(4);
        for (&char, &count) in all_counts.iter() {
            if index[char as usize] >= 0 {
                freqs[index[char as usize] as usize] += count;
            } else {
                let charset = match dna_ambiguous_chars.get(&char) {
                    Some(set) => set,
                    None => {
                        warn!(
                            "Unknown character {} encountered, treating it as ambiguous.",
                            char
                        );
                        dna_ambiguous_chars.get(&AMBIG).unwrap()
                    }
                };
                let num = charset.len() as f64;
                for &char in charset {
                    freqs[index[char as usize] as usize] += count / num;
                }
            }
        }
        freqs
    }
}

pub fn nucleotide_index() -> [i32; 255] {
    let mut index = [-1_i32; 255];
    for (i, char) in charify(NUCLEOTIDES_STR).into_iter().enumerate() {
        index[char as usize] = i as i32;
        index[char.to_ascii_lowercase() as usize] = i as i32;
    }
    index
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
