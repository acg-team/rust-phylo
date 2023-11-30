use std::collections::HashMap;
use std::ops::Div;

use anyhow::bail;
use log::warn;
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelInfo};
use crate::likelihood::LikelihoodCostFunction;
use crate::sequences::{charify, NUCLEOTIDES_STR};
use crate::substitution_models::{
    FreqVector, ParsimonyModel, SubstMatrix, SubstitutionLikelihoodCost, SubstitutionModel,
    SubstitutionModelInfo,
};
use crate::{Result, Rounding};

pub type DNASubstModel = SubstitutionModel<4>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, 4>;
pub type DNASubstModelInfo = SubstitutionModelInfo<4>;

mod hky;
mod jc69;
mod k80;
mod tn93;
pub use hky::*;
pub use jc69::*;
pub use k80::*;
pub use tn93::*;

struct GtrParams<'a> {
    pi: &'a FreqVector,
    rtc: f64,
    rta: f64,
    rtg: f64,
    rca: f64,
    rcg: f64,
    rag: f64,
}

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

pub fn gtr(model_params: &[f64]) -> Result<DNASubstModel> {
    if model_params.len() != 10 {
        bail!(
            "{} parameters for the GTR model, expected 10, got {}",
            if model_params.len() < 10 {
                "Not enough"
            } else {
                "Too many"
            },
            model_params.len()
        );
    }
    let pi = FreqVector::from_column_slice(&[
        model_params[0],
        model_params[1],
        model_params[2],
        model_params[3],
    ]);
    if pi.sum() != 1.0 {
        bail!("The equilibrium frequencies provided do not sum up to 1.");
    }
    let gtr_params = &GtrParams {
        pi: &pi,
        rtc: model_params[4],
        rta: model_params[5],
        rtg: model_params[6],
        rca: model_params[7],
        rcg: model_params[8],
        rag: model_params[9],
    };

    Ok(make_dna_model(
        model_params[0..10].to_vec(),
        gtr_q(gtr_params),
        pi,
    ))
}

fn gtr_q(gtr: &GtrParams) -> SubstMatrix {
    let ft = gtr.pi[0];
    let fc = gtr.pi[1];
    let fa = gtr.pi[2];
    let fg = gtr.pi[3];
    let total = (gtr.rtc * fc + gtr.rta * fa + gtr.rtg * fg) * ft
        + (gtr.rtc * ft + gtr.rca * fa + gtr.rcg * fg) * fc
        + (gtr.rta * ft + gtr.rca * fc + gtr.rag * fg) * fa
        + (gtr.rtg * ft + gtr.rcg * fc + gtr.rag * fa) * fg;
    SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -(gtr.rtc * fc + gtr.rta * fa + gtr.rtg * fg),
            gtr.rtc * fc,
            gtr.rta * fa,
            gtr.rtg * fg,
            gtr.rtc * ft,
            -(gtr.rtc * ft + gtr.rca * fa + gtr.rcg * fg),
            gtr.rca * fa,
            gtr.rcg * fg,
            gtr.rta * ft,
            gtr.rca * fc,
            -(gtr.rta * ft + gtr.rca * fc + gtr.rag * fg),
            gtr.rag * fg,
            gtr.rtg * ft,
            gtr.rcg * fc,
            gtr.rag * fa,
            -(gtr.rtg * ft + gtr.rcg * fc + gtr.rag * fa),
        ],
    )
    .div(total)
}
