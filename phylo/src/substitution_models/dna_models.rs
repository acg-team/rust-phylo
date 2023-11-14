use std::collections::HashMap;

use anyhow::bail;
use log::{info, warn};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvolutionaryModel;
use crate::sequences::{charify, NUCLEOTIDES_STR};
use crate::substitution_models::{FreqVector, SubstMatrix, SubstitutionModel};
use crate::{Result, Rounding};

pub type DNASubstModel = SubstitutionModel<4>;

impl EvolutionaryModel<4> for DNASubstModel {
    fn new(model_name: &str, model_params: &[f64], normalise: bool) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let (q, pi) = match model_name.to_uppercase().as_str() {
            "JC69" => jc69(model_params)?,
            "K80" => k80(model_params)?,
            "TN93" => tn93(model_params)?,
            "HKY" => hky(model_params)?,
            "GTR" => gtr(model_params)?,
            _ => bail!("Unknown DNA model requested."),
        };
        let mut model = DNASubstModel {
            index: nucleotide_index(),
            q,
            pi,
        };
        if normalise {
            model.normalise();
        }
        Ok(model)
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        self.get_p(time)
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.get_rate(i, j)
    }

    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        self.generate_scorings(times, zero_diag, rounding)
    }

    fn normalise(&mut self) {
        self.normalise()
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix(time, rounding)
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

pub fn nucleotide_index() -> [i32; 255] {
    let mut index = [-1_i32; 255];
    for (i, char) in charify(NUCLEOTIDES_STR).into_iter().enumerate() {
        index[char as usize] = i as i32;
        index[char.to_ascii_lowercase() as usize] = i as i32;
    }
    index
}

pub fn jc69(model_params: &[f64]) -> Result<(SubstMatrix, FreqVector)> {
    if model_params.is_empty() {
        warn!("Too many values provided for JC69 (>0).");
        warn!("Provided values will be ignored.");
    }
    Ok((
        SubstMatrix::from_row_slice(4, 4, &JC69_ARR),
        FreqVector::from_column_slice(JC69_PI_ARR.as_slice()),
    ))
}

pub fn k80(model_params: &[f64]) -> Result<(SubstMatrix, FreqVector)> {
    let (alpha, beta) = if model_params.len() < 2 {
        warn!("Too few values provided for K80, required 2 values, alpha and beta.");
        warn!("Falling back to default values.");
        (2.0, 1.0)
    } else {
        if model_params.len() > 2 {
            warn!("Too many values provided for K80, required 2 values, alpha and beta.");
            warn!("Will only use the first two values provided.");
        }
        (model_params[0], model_params[1])
    };
    info!("Setting up k80 with alpha = {}, beta = {}", alpha, beta);
    Ok((
        SubstMatrix::from_row_slice(
            4,
            4,
            [
                -1.0,
                alpha / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                alpha / (alpha + 2.0 * beta),
                -1.0,
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                -1.0,
                alpha / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                alpha / (alpha + 2.0 * beta),
                -1.0,
            ]
            .as_slice(),
        )
        .transpose(),
        FreqVector::from_column_slice(JC69_PI_ARR.as_slice()),
    ))
}

pub fn hky(model_params: &[f64]) -> Result<(SubstMatrix, FreqVector)> {
    if model_params.len() != 5 {
        bail!(
            "{} parameters for the hky model, expected 5, got {}",
            if model_params.len() < 10 {
                "Not enough"
            } else {
                "Too many"
            },
            model_params.len()
        );
    }
    let f_t = model_params[0];
    let f_c = model_params[1];
    let f_a = model_params[2];
    let f_g = model_params[3];
    let a1 = model_params[4];
    let a2 = model_params[4];
    let b = 1.0;
    info!("Setting up hky with alpha = {}", a1);
    if (f_t + f_c + f_a + f_g) != 1.0 {
        bail!("The equilibrium frequencies provided do not sum up to 1.");
    }
    let q = tn93_matrix(f_t, f_c, f_a, f_g, a1, a2, b);
    Ok((q, FreqVector::from_column_slice(&[f_t, f_c, f_a, f_g])))
}

pub fn tn93(model_params: &[f64]) -> Result<(SubstMatrix, FreqVector)> {
    if model_params.len() != 7 {
        bail!(
            "{} parameters for the tn93 model, expected 7, got {}",
            if model_params.len() < 10 {
                "Not enough"
            } else {
                "Too many"
            },
            model_params.len()
        );
    }
    let f_t = model_params[0];
    let f_c = model_params[1];
    let f_a = model_params[2];
    let f_g = model_params[3];
    let a1 = model_params[4];
    let a2 = model_params[5];
    let b = model_params[6];
    info!(
        "Setting up tn93 with alpha1 = {}, alpha2 = {}, beta = {}",
        a1, a2, b
    );
    if (f_t + f_c + f_a + f_g) != 1.0 {
        bail!("The equilibrium frequencies provided do not sum up to 1.");
    }
    let q = tn93_matrix(f_t, f_c, f_a, f_g, a1, a2, b);
    Ok((q, FreqVector::from_column_slice(&[f_t, f_c, f_a, f_g])))
}

fn tn93_matrix(f_t: f64, f_c: f64, f_a: f64, f_g: f64, a1: f64, a2: f64, b: f64) -> SubstMatrix {
    let mut q = SubstMatrix::from_row_slice(
        4,
        4,
        &[
            0.0,
            a1 * f_c,
            b * f_a,
            b * f_g,
            a1 * f_t,
            0.0,
            b * f_a,
            b * f_g,
            b * f_t,
            b * f_c,
            0.0,
            a2 * f_g,
            b * f_t,
            b * f_c,
            a2 * f_a,
            0.0,
        ],
    );
    for i in 0..4 {
        q[(i, i)] = -q.row(i).sum();
    }
    q
}

pub(crate) fn gtr(model_params: &[f64]) -> Result<(SubstMatrix, FreqVector)> {
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
    let f_t = model_params[0];
    let f_c = model_params[1];
    let f_a = model_params[2];
    let f_g = model_params[3];
    let r_tc = model_params[4];
    let r_ta = model_params[5];
    let r_tg = model_params[6];
    let r_ca = model_params[7];
    let r_cg = model_params[8];
    let r_ag = model_params[9];
    if (f_t + f_c + f_a + f_g) != 1.0 {
        bail!("The equilibrium frequencies provided do not sum up to 1.");
    }
  
    let mut q = SubstMatrix::from_row_slice(
        4,
        4,
        &[
            0.0,
            r_tc * f_c,
            r_ta * f_a,
            r_tg * f_g,
            r_tc * f_t,
            0.0,
            r_ca * f_a,
            r_cg * f_g,
            r_ta * f_t,
            r_ca * f_c,
            0.0,
            r_ag * f_g,
            r_tg * f_t,
            r_cg * f_c,
            r_ag * f_a,
            0.0,
        ],
    );
    for i in 0..4 {
        q[(i, i)] = -q.row(i).sum();
    }
    Ok((q, FreqVector::from_column_slice(&[f_t, f_c, f_a, f_g])))
}

const JC69_ARR: [f64; 16] = [
    -1.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    -1.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    -1.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
    -1.0,
];

const JC69_PI_ARR: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
