use std::collections::HashMap;

use anyhow::bail;
use log::{info, warn};
use nalgebra::DVector;
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvolutionaryModel;
use crate::sequences::{charify, NUCLEOTIDES_STR};
use crate::substitution_models::{FreqVector, SubstMatrix, SubstitutionModel};
use crate::Result;

type DNASubstMatrix = SubstMatrix<4>;
type DNAFreqVector = FreqVector<4>;

pub type DNASubstModel = SubstitutionModel<4>;

impl EvolutionaryModel<4> for DNASubstModel {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let q: SubstMatrix<4>;
        let pi: FreqVector<4>;
        match model_name.to_uppercase().as_str() {
            "JC69" => (q, pi) = jc69(model_params)?,
            "K80" => (q, pi) = k80(model_params)?,
            "TN93" => (q, pi) = tn93(model_params)?,
            "GTR" => (q, pi) = gtr(model_params)?,
            _ => bail!("Unknown DNA model requested."),
        }
        let model = DNASubstModel {
            index: nucleotide_index(),
            q,
            pi,
        };
        Ok(model)
    }

    fn get_p(&self, time: f64) -> SubstMatrix<4> {
        self.get_p(time)
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        self.get_rate(i, j)
    }

    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix<4>, f64)> {
        self.generate_scorings(times, zero_diag, rounded)
    }

    fn normalise(&mut self) {
        self.normalise()
    }

    fn get_scoring_matrix(&self, time: f64, rounded: bool) -> (SubstMatrix<4>, f64) {
        self.get_scoring_matrix(time, rounded)
    }

    fn get_stationary_distribution(&self) -> &FreqVector<4> {
        self.get_stationary_distribution()
    }

    fn get_char_probability(&self, char: u8) -> DVector<f64> {
        let mut vec = DVector::<f64>::zeros(4);
        if NUCLEOTIDES_STR.contains(char as char) {
            vec[self.index[char as usize] as usize] = 1.0;
        } else {
            vec = DVector::from_column_slice(self.get_stationary_distribution().as_slice());
            match char {
                b'-' => {}
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
                _ => warn!("Unknown character {} encountered.", char),
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

pub fn jc69(model_params: &[f64]) -> Result<(DNASubstMatrix, DNAFreqVector)> {
    if model_params.is_empty() {
        warn!("Too many values provided for JC69 (>0).");
        warn!("Provided values will be ignored.");
    }
    Ok((
        DNASubstMatrix::from(JC69_ARR).transpose(),
        DNAFreqVector::from(JC69_PI_ARR),
    ))
}

pub fn k80(model_params: &[f64]) -> Result<(DNASubstMatrix, DNAFreqVector)> {
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
        DNASubstMatrix::from([
            [
                -1.0,
                alpha / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
            ],
            [
                alpha / (alpha + 2.0 * beta),
                -1.0,
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
            ],
            [
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                -1.0,
                alpha / (alpha + 2.0 * beta),
            ],
            [
                beta / (alpha + 2.0 * beta),
                beta / (alpha + 2.0 * beta),
                alpha / (alpha + 2.0 * beta),
                -1.0,
            ],
        ])
        .transpose(),
        DNAFreqVector::from(JC69_PI_ARR),
    ))
}

pub fn tn93(model_params: &[f64]) -> Result<(DNASubstMatrix, DNAFreqVector)> {
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
    let mut q = DNASubstMatrix::from([
        [0.0, a1 * f_c, b * f_a, b * f_g],
        [a1 * f_t, 0.0, b * f_a, b * f_g],
        [b * f_t, b * f_c, 0.0, a2 * f_g],
        [b * f_t, b * f_c, a2 * f_a, 0.0],
    ]);
    q.transpose_mut();
    for i in 0..4 {
        q[(i, i)] = -q.row(i).sum();
    }
    Ok((q, DNAFreqVector::from([f_t, f_c, f_a, f_g])))
}

pub(crate) fn gtr(model_params: &[f64]) -> Result<(DNASubstMatrix, DNAFreqVector)> {
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
    let mut q = DNASubstMatrix::from([
        [0.0, r_tc * f_c, r_ta * f_a, r_tg * f_g],
        [r_tc * f_t, 0.0, r_ca * f_a, r_cg * f_g],
        [r_ta * f_t, r_ca * f_c, 0.0, r_ag * f_g],
        [r_tg * f_t, r_cg * f_c, r_ag * f_a, 0.0],
    ]);
    q.transpose_mut();
    for i in 0..4 {
        q[(i, i)] = -q.row(i).sum();
    }
    Ok((q, DNAFreqVector::from([f_t, f_c, f_a, f_g])))
}

const JC69_ARR: [[f64; 4]; 4] = [
    [-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, -1.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0],
];

const JC69_PI_ARR: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
