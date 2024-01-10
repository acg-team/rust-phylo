use std::collections::{HashMap, HashSet};

use anyhow::bail;
use log::warn;
use map_macro::{hash_map, hash_set};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvolutionaryModel;
use crate::likelihood::LikelihoodCostFunction;
use crate::sequences::{charify, dna_alphabet, AMBIG, NUCLEOTIDES_STR};
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

fn dna_ambiguous_chars() -> HashMap<u8, HashSet<u8>> {
    hash_map! {
        b'V' => hash_set! {b'C', b'A', b'G'},
        b'v' => hash_set! {b'C', b'A', b'G'},
        b'D' => hash_set! {b'T', b'A', b'G'},
        b'd' => hash_set! {b'T', b'A', b'G'},
        b'B' => hash_set! {b'T', b'C', b'G'},
        b'b' => hash_set! {b'T', b'C', b'G'},
        b'H' => hash_set! {b'T', b'C' ,b'A'},
        b'h' => hash_set! {b'T', b'C', b'A'},
        b'M' => hash_set! {b'A', b'C'},
        b'm' => hash_set! {b'A', b'C'},
        b'R' => hash_set! {b'A', b'G'},
        b'r' => hash_set! {b'A', b'G'},
        b'W' => hash_set! {b'A', b'T'},
        b'w' => hash_set! {b'A', b'T'},
        b'S' => hash_set! {b'C', b'G'},
        b's' => hash_set! {b'C', b'G'},
        b'Y' => hash_set! {b'C', b'T'},
        b'y' => hash_set! {b'C', b'T'},
        b'K' => hash_set! {b'G', b'T'},
        b'k' => hash_set! {b'G', b'T'},
        b'X' => hash_set! {b'A', b'C', b'G', b'T'},
        b'x' => hash_set! {b'A', b'C', b'G', b'T'},
        b'N' => hash_set! {b'A', b'C', b'G', b'T'},
        b'n' => hash_set! {b'A', b'C', b'G', b'T'},
    }
}

fn make_dna_model(params: DNASubstParams, q: SubstMatrix) -> DNASubstModel {
    let pi = params.pi.clone();
    DNASubstModel {
        params: SubstParams::DNA(params),
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
        let dna_ambiguous_chars = dna_ambiguous_chars();
        let dna_char_set: HashSet<u8> =
            HashSet::from_iter(NUCLEOTIDES_STR.as_bytes().iter().cloned());
        if NUCLEOTIDES_STR.contains(char as char) {
            probs[self.index[char as usize] as usize] = 1.0;
        } else {
            probs = FreqVector::from_column_slice(self.get_stationary_distribution().as_slice());
            let other = dna_ambiguous_chars.get(&char);
            match other {
                Some(other) => {
                    let difference = dna_char_set.difference(other);
                    for &char in difference {
                        probs[self.index[char as usize] as usize] = 0.0;
                    }
                }
                None => {
                    warn!("Unknown character {} encountered, treating it as X.", char);
                    probs =
                        FreqVector::from_column_slice(self.get_stationary_distribution().as_slice())
                }
            }
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
        for &char in NUCLEOTIDES_STR.as_bytes() {
            if freqs[index[char as usize] as usize] == 0.0 {
                freqs[index[char as usize] as usize] += 1.0;
                total += 1.0;
            }
        }
        freqs.map(|x| x / total)
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

#[cfg(test)]
mod dna_optimisation_tests;
