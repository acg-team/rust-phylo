use crate::sequences::{charify, NUCLEOTIDES_STR};
use crate::substitution_models::{FreqVector, SubstMatrix};
use crate::Result;
use anyhow::bail;
use log::{info, warn};

type DNASubstMatrix = SubstMatrix<4>;
type DNAFreqVector = FreqVector<4>;

pub fn nucleotide_index() -> [i32; 255] {
    let mut index = [-1 as i32; 255];
    for (i, char) in charify(NUCLEOTIDES_STR).into_iter().enumerate() {
        index[char as usize] = i as i32;
        index[char.to_ascii_lowercase() as usize] = i as i32;
    }
    index
}

pub fn jc69(model_params: &[f64]) -> Result<(DNASubstMatrix, DNAFreqVector)> {
    if model_params.len() > 0 {
        warn!("Too many values provided for JC69 (>0).");
        warn!("Provided values will be ignored.");
    }
    Ok((
        DNASubstMatrix::from(JC69_ARR),
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
        ]),
        DNAFreqVector::from(JC69_PI_ARR),
    ))
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
    Ok((
        DNASubstMatrix::from([
            [
                -(r_tc * f_c + r_ta * f_a + r_tg * f_g),
                r_tc * f_c,
                r_ta * f_a,
                r_tg * f_g,
            ],
            [
                r_tc * f_t,
                -(r_tc * f_t + r_ca * f_a + r_cg * f_g),
                r_ca * f_a,
                r_cg * f_g,
            ],
            [
                r_ta * f_t,
                r_ca * f_c,
                -(r_ta * f_t + r_ca * f_c + r_ag * f_g),
                r_ag * f_g,
            ],
            [
                r_tg * f_t,
                r_cg * f_c,
                r_ag * f_a,
                -(r_tg * f_t + r_cg * f_c + r_ag * f_a),
            ],
        ]),
        DNAFreqVector::from([f_t, f_c, f_a, f_g]),
    ))
}

const JC69_ARR: [[f64; 4]; 4] = [
    [-1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, -1.0, 1.0 / 3.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, -1.0, 1.0 / 3.0],
    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, -1.0],
];

const JC69_PI_ARR: [f64; 4] = [0.25, 0.25, 0.25, 0.25];
