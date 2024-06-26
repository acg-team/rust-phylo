use std::collections::HashMap;
use std::fmt::Display;

use anyhow::bail;
use lazy_static::lazy_static;
use log::info;
use ordered_float::OrderedFloat;

use crate::evolutionary_models::{EvolutionaryModel, EvolutionaryModelParameters};
use crate::likelihood::LikelihoodCostFunction;
use crate::sequences::{charify, GAP, NUCLEOTIDES_STR};
use crate::substitution_models::{
    FreqVector, ParsimonyModel, SubstMatrix, SubstParams, SubstitutionLikelihoodCost,
    SubstitutionModel, SubstitutionModelInfo,
};
use crate::{make_freqs, Result, Rounding};

pub type DNASubstModel = SubstitutionModel<4>;
pub type DNALikelihoodCost<'a> = SubstitutionLikelihoodCost<'a, 4>;
pub type DNASubstModelInfo = SubstitutionModelInfo<4>;

mod dna_substitution_parameters;
pub use dna_substitution_parameters::*;
pub mod dna_model_optimiser;
pub(crate) use common_dna_models::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DNAModelType {
    JC69,
    K80,
    HKY,
    TN93,
    GTR,
}

impl Display for DNAModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DNAModelType::JC69 => write!(f, "JC69"),
            DNAModelType::K80 => write!(f, "K80"),
            DNAModelType::HKY => write!(f, "HKY"),
            DNAModelType::TN93 => write!(f, "TN93"),
            DNAModelType::GTR => write!(f, "GTR"),
        }
    }
}

lazy_static! {
    pub static ref DNA_GAP_SETS: Vec<FreqVector> = {
        let index = &NUCLEOTIDE_INDEX;
        let mut map = Vec::<FreqVector>::new();
        let mut x_set = FreqVector::from_element(5, 1.0 / 4.0);
        x_set.fill_row(4, 0.0);
        map.resize(255, x_set.clone());
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8 as char;
            elem.set_column(
                0,
                &match char {
                    'T' | 't' | 'C' | 'c' | 'A' | 'a' | 'G' | 'g' => {
                        let mut set = FreqVector::zeros(5);
                        set.fill_row(index[char as usize] as usize, 1.0);
                        set
                    }
                    'V' | 'v' => {
                        make_freqs!(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
                    }
                    'D' | 'd' => {
                        make_freqs!(&[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
                    }
                    'B' | 'b' => {
                        make_freqs!(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0, 0.0])
                    }
                    'H' | 'h' => {
                        make_freqs!(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 0.0])
                    }
                    'M' | 'm' => {
                        make_freqs!(&[0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0])
                    }
                    'R' | 'r' => {
                        make_freqs!(&[0.0, 0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0])
                    }
                    'W' | 'w' => {
                        make_freqs!(&[1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0, 0.0])
                    }
                    'S' | 's' => {
                        make_freqs!(&[0.0, 1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0])
                    }
                    'Y' | 'y' => {
                        make_freqs!(&[1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0, 0.0])
                    }
                    'K' | 'k' => {
                        make_freqs!(&[1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0, 0.0])
                    }
                    '-' => make_freqs!(&[0.0, 0.0, 0.0, 0.0, 1.0]),
                    _ => continue,
                },
            );
        }
        map
    };
    pub static ref DNA_SETS: Vec<FreqVector> = {
        let index = &NUCLEOTIDE_INDEX;
        let mut map = Vec::<FreqVector>::new();
        map.resize(255, FreqVector::from_element(4, 1.0 / 4.0));
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8 as char;
            elem.set_column(
                0,
                &match char {
                    'T' | 't' | 'C' | 'c' | 'A' | 'a' | 'G' | 'g' => {
                        let mut set = FreqVector::zeros(4);
                        set.fill_row(index[char as usize] as usize, 1.0);
                        set
                    }
                    'V' | 'v' => {
                        make_freqs!(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
                    }
                    'D' | 'd' => {
                        make_freqs!(&[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0])
                    }
                    'B' | 'b' => {
                        make_freqs!(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
                    }
                    'H' | 'h' => {
                        make_freqs!(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
                    }
                    'M' | 'm' => make_freqs!(&[0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0]),
                    'R' | 'r' => make_freqs!(&[0.0, 0.0, 1.0 / 2.0, 1.0 / 2.0]),
                    'W' | 'w' => make_freqs!(&[1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0]),
                    'S' | 's' => make_freqs!(&[0.0, 1.0 / 2.0, 0.0, 1.0 / 2.0]),
                    'Y' | 'y' => make_freqs!(&[1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0]),
                    'K' | 'k' => make_freqs!(&[1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0]),
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
        index[GAP as usize] = 4;
        index
    };
}

pub(crate) fn make_dna_model(params: DNASubstParams) -> DNASubstModel {
    info!(
        "Setting up {} with parameters: {}",
        params.model_type, params
    );
    let pi = params.pi.clone();
    let q = match params.model_type {
        DNAModelType::JC69 => common_dna_models::jc69_q(),
        DNAModelType::K80 => common_dna_models::k80_q(&params),
        DNAModelType::HKY => tn93_q(&params),
        DNAModelType::TN93 => tn93_q(&params),
        DNAModelType::GTR => gtr_q(&params),
    };
    DNASubstModel {
        params: SubstParams::DNA(params),
        index: *NUCLEOTIDE_INDEX,
        q,
        pi,
    }
}

impl EvolutionaryModel<4> for DNASubstModel {
    fn new(model_name: &str, params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let model_type = match model_name.to_uppercase().as_str() {
            "JC69" => DNAModelType::JC69,
            "K80" => DNAModelType::K80,
            "HKY" => DNAModelType::HKY,
            "TN93" => DNAModelType::TN93,
            "GTR" => DNAModelType::GTR,
            _ => bail!("Unknown DNA model requested."),
        };
        let params = DNASubstParams::new(&model_type, params)?;
        Ok(make_dna_model(params))
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
        let all_counts = self.info.get_counts();
        let mut total = all_counts.values().sum::<f64>();
        let index = &NUCLEOTIDE_INDEX;
        let mut freqs = FreqVector::zeros(4);
        for (&char, &count) in all_counts.iter() {
            freqs += &DNA_SETS[char as usize].scale(count);
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

pub(super) mod common_dna_models {
    use std::ops::Div;

    use anyhow::bail;
    use log::warn;

    use crate::substitution_models::{
        dna_models::{DNAModelType, DNASubstParams},
        FreqVector, SubstMatrix,
    };
    use crate::{make_freqs, Result};

    fn make_pi(pi_array: &[f64]) -> Result<FreqVector> {
        let pi = make_freqs!(pi_array);
        debug_assert!(
            pi.len() == 4,
            "There have to be 4 equilibrium frequencies for DNA models."
        );
        if pi.sum() != 1.0 {
            bail!("The equilibrium frequencies provided do not sum up to 1.");
        }
        Ok(pi)
    }

    fn make_q(q_array: &[f64]) -> SubstMatrix {
        SubstMatrix::from_row_slice(4, 4, q_array)
    }

    pub(crate) fn jc69_params(params: &[f64]) -> Result<DNASubstParams> {
        if !params.is_empty() {
            warn!("Too many values provided for JC69, average rate is fixed at 1.0.");
        }
        Ok(DNASubstParams {
            model_type: DNAModelType::JC69,
            pi: make_pi(&[0.25; 4])?,
            rtc: 1.0,
            rta: 1.0,
            rtg: 1.0,
            rca: 1.0,
            rcg: 1.0,
            rag: 1.0,
        })
    }

    pub(crate) fn jc69_q() -> SubstMatrix {
        let r = 1.0 / 3.0;
        make_q(&[-1.0, r, r, r, r, -1.0, r, r, r, r, -1.0, r, r, r, r, -1.0])
    }

    pub(crate) fn k80_params(params: &[f64]) -> Result<DNASubstParams> {
        let (a, b) = if params.is_empty() {
            warn!(
                "Too few values provided for K80, required 1 or 2 values, kappa or alpha and beta."
            );
            warn!("Falling back to default values.");
            (2.0, 1.0)
        } else if params.len() == 1 {
            (params[0], 1.0)
        } else if params.len() == 2 {
            (params[0], params[1])
        } else {
            warn!("Too many values provided for K80, required 2 values, alpha and beta.");
            warn!("Will only use the first two values provided.");
            (params[0], params[1])
        };
        Ok(DNASubstParams {
            model_type: DNAModelType::K80,
            pi: make_pi(&[0.25; 4])?,
            rtc: a,
            rta: b,
            rtg: b,
            rca: b,
            rcg: b,
            rag: a,
        })
    }

    pub(crate) fn k80_q(p: &DNASubstParams) -> SubstMatrix {
        let a = p.rtc;
        let b = p.rta;
        let total = a + 2.0 * b;
        make_q(&[
            -(a + 2.0 * b),
            a,
            b,
            b,
            a,
            -(a + 2.0 * b),
            b,
            b,
            b,
            b,
            -(a + 2.0 * b),
            a,
            b,
            b,
            a,
            -(a + 2.0 * b),
        ])
        .div(total)
    }

    pub(crate) fn hky_params(params: &[f64]) -> Result<DNASubstParams> {
        if params.len() < 4 {
            bail!(
                "Too few parameters for the hky model, expected at least 4, got {}",
                params.len()
            );
        }
        let (a, b) = if params.len() == 4 {
            warn!("Too few values provided for HKY, required pi and 1 or 2 values, kappa or alpha and beta.");
            warn!("Falling back to default values.");
            (2.0, 1.0)
        } else if params.len() == 5 {
            (params[4], 1.0)
        } else if params.len() == 6 {
            (params[4], params[5])
        } else {
            warn!("Too many values provided for HKY, required pi and 1 or 2 values, kappa or alpha and beta.");
            warn!("Will only use the first values provided.");
            (params[4], params[5])
        };
        Ok(DNASubstParams {
            model_type: DNAModelType::HKY,
            pi: make_pi(&[params[0], params[1], params[2], params[3]])?,
            rtc: a,
            rta: b,
            rtg: b,
            rca: b,
            rcg: b,
            rag: a,
        })
    }

    pub fn tn93_params(params: &[f64]) -> Result<DNASubstParams> {
        if params.len() != 7 {
            bail!(
                "{} parameters for the tn93 model, expected 7, got {}",
                if params.len() < 7 {
                    "Not enough"
                } else {
                    "Too many"
                },
                params.len()
            );
        }
        let a1 = params[4];
        let a2 = params[5];
        let b = params[6];
        Ok(DNASubstParams {
            model_type: DNAModelType::TN93,
            pi: make_pi(&[params[0], params[1], params[2], params[3]])?,
            rtc: a1,
            rta: b,
            rtg: b,
            rca: b,
            rcg: b,
            rag: a2,
        })
    }

    pub(crate) fn tn93_q(p: &DNASubstParams) -> SubstMatrix {
        let ft = p.pi[0];
        let fc = p.pi[1];
        let fa = p.pi[2];
        let fg = p.pi[3];
        let a1 = p.rtc;
        let a2 = p.rag;
        let b = p.rta;
        let total = (a1 * fc + b * (fa + fg)) * ft
            + (a1 * ft + b * (fa + fg)) * fc
            + (b * (ft + fc) + a2 * fg) * fa
            + (b * (ft + fc) + a2 * fa) * fg;
        SubstMatrix::from_row_slice(
            4,
            4,
            &[
                -(a1 * fc + b * (fa + fg)),
                a1 * fc,
                b * fa,
                b * fg,
                a1 * ft,
                -(a1 * ft + b * (fa + fg)),
                b * fa,
                b * fg,
                b * ft,
                b * fc,
                -(b * (ft + fc) + a2 * fg),
                a2 * fg,
                b * ft,
                b * fc,
                a2 * fa,
                -(b * (ft + fc) + a2 * fa),
            ],
        )
        .div(total)
    }

    pub fn gtr_params(params: &[f64]) -> Result<DNASubstParams> {
        if params.len() != 10 {
            bail!(
                "{} parameters for the GTR model, expected 10, got {}",
                if params.len() < 10 {
                    "Not enough"
                } else {
                    "Too many"
                },
                params.len()
            );
        }

        let gtr_params = DNASubstParams {
            model_type: DNAModelType::GTR,
            pi: make_pi(&[params[0], params[1], params[2], params[3]])?,
            rtc: params[4],
            rta: params[5],
            rtg: params[6],
            rca: params[7],
            rcg: params[8],
            rag: params[9],
        };

        Ok(gtr_params)
    }

    pub(crate) fn gtr_q(gtr: &DNASubstParams) -> SubstMatrix {
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
}

#[cfg(test)]
mod dna_optimisation_tests;
