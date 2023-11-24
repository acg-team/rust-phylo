use std::collections::HashMap;

use anyhow::bail;
use log::{info, warn};
use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvolutionaryModel;
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

struct GtrParams<'a> {
    pi: &'a FreqVector,
    rtc: f64,
    rta: f64,
    rtg: f64,
    rca: f64,
    rcg: f64,
    rag: f64,
}

struct TN93Params<'a> {
    pi: &'a FreqVector,
    a1: f64,
    a2: f64,
    b: f64,
}

impl EvolutionaryModel<4> for DNASubstModel {
    fn new(model_name: &str, model_params: &[f64], normalise: bool) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        let (params, q, pi) = match model_name.to_uppercase().as_str() {
            "JC69" => jc69(model_params)?,
            "K80" => k80(model_params)?,
            "TN93" => tn93(model_params)?,
            "HKY" => hky(model_params)?,
            "GTR" => gtr(model_params)?,
            _ => bail!("Unknown DNA model requested."),
        };
        let mut model = DNASubstModel {
            params,
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

    fn normalise(&mut self) {
        self.normalise()
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
        self.compute_log_likelihood(model, tmp_info)
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

pub fn jc69(model_params: &[f64]) -> Result<(Vec<f64>, SubstMatrix, FreqVector)> {
    let mu = if model_params.is_empty() {
        warn!("Too few values provided for JC69, required 1 value, mu.");
        warn!("Falling back to default values.");
        1.0
    } else if model_params.len() == 1 {
        model_params[0]
    } else {
        warn!("Too many values provided for JC69, required 1 value, mu.");
        warn!("Will only use the first value provided.");
        model_params[0]
    };
    Ok((
        vec![mu],
        jc69_q(mu),
        FreqVector::from_column_slice(&[0.25; 4]),
    ))
}

pub fn jc69_q(mu: f64) -> SubstMatrix {
    let rate = mu / 4.0;
    SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -3.0 * rate,
            rate,
            rate,
            rate,
            rate,
            -3.0 * rate,
            rate,
            rate,
            rate,
            rate,
            -3.0 * rate,
            rate,
            rate,
            rate,
            rate,
            -3.0 * rate,
        ],
    )
}

pub fn k80(model_params: &[f64]) -> Result<(Vec<f64>, SubstMatrix, FreqVector)> {
    let (alpha, beta) = if model_params.is_empty() {
        warn!("Too few values provided for K80, required 1 or 2 values, kappa or alpha and beta.");
        warn!("Falling back to default values.");
        (2.0, 1.0)
    } else if model_params.len() == 1 {
        (model_params[0], 1.0)
    } else if model_params.len() == 2 {
        (model_params[0], model_params[1])
    } else {
        warn!("Too many values provided for K80, required 2 values, alpha and beta.");
        warn!("Will only use the first two values provided.");
        (model_params[0], model_params[1])
    };
    info!("Setting up k80 with alpha = {}, beta = {}", alpha, beta);
    Ok((
        vec![alpha, beta],
        k80_q(alpha, beta),
        FreqVector::from_column_slice(&[0.25; 4]),
    ))
}

pub fn k80_q(alpha: f64, beta: f64) -> SubstMatrix {
    SubstMatrix::from_column_slice(
        4,
        4,
        &[
            -(alpha + 2.0 * beta),
            alpha,
            beta,
            beta,
            alpha,
            -(alpha + 2.0 * beta),
            beta,
            beta,
            beta,
            beta,
            -(alpha + 2.0 * beta),
            alpha,
            beta,
            beta,
            alpha,
            -(alpha + 2.0 * beta),
        ],
    )
}

pub fn hky(model_params: &[f64]) -> Result<(Vec<f64>, SubstMatrix, FreqVector)> {
    if model_params.len() != 5 {
        bail!(
            "{} parameters for the hky model, expected 5, got {}",
            if model_params.len() < 5 {
                "Not enough"
            } else {
                "Too many"
            },
            model_params.len()
        );
    }
    let pi = make_pi(&model_params[0..4])?;
    let hky_params = &TN93Params {
        pi: &pi,
        a1: model_params[4],
        a2: model_params[4],
        b: 1.0,
    };
    info!("Setting up hky with alpha = {}", hky_params.a1);
    Ok((model_params[0..5].to_vec(), tn93_q(hky_params), pi))
}

pub fn tn93(model_params: &[f64]) -> Result<(Vec<f64>, SubstMatrix, FreqVector)> {
    if model_params.len() != 7 {
        bail!(
            "{} parameters for the tn93 model, expected 7, got {}",
            if model_params.len() < 7 {
                "Not enough"
            } else {
                "Too many"
            },
            model_params.len()
        );
    }
    let pi = make_pi(&model_params[0..4])?;
    let tn93_params = &TN93Params {
        pi: &pi,
        a1: model_params[4],
        a2: model_params[5],
        b: model_params[6],
    };
    info!(
        "Setting up tn93 with alpha1 = {}, alpha2 = {}, beta = {}",
        tn93_params.a1, tn93_params.a2, tn93_params.b
    );
    Ok((model_params[0..7].to_vec(), tn93_q(tn93_params), pi))
}

fn tn93_q(p: &TN93Params) -> SubstMatrix {
    let ft = p.pi[0];
    let fc = p.pi[1];
    let fa = p.pi[2];
    let fg = p.pi[3];
    SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -(p.a1 * fc + p.b * (fa + fg)),
            p.a1 * fc,
            p.b * fa,
            p.b * fg,
            p.a1 * ft,
            -(p.a1 * ft + p.b * (fa + fg)),
            p.b * fa,
            p.b * fg,
            p.b * ft,
            p.b * fc,
            -(p.b * (ft + fc) + p.a2 * fg),
            p.a2 * fg,
            p.b * ft,
            p.b * fc,
            p.a2 * fa,
            -(p.b * (ft + fc) + p.a2 * fa),
        ],
    )
}

pub fn gtr(model_params: &[f64]) -> Result<(Vec<f64>, SubstMatrix, FreqVector)> {
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

    Ok((model_params[0..10].to_vec(), gtr_q(gtr_params), pi))
}

fn gtr_q(gtr: &GtrParams) -> SubstMatrix {
    SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -(gtr.rtc * gtr.pi[1] + gtr.rta * gtr.pi[2] + gtr.rtg * gtr.pi[3]),
            gtr.rtc * gtr.pi[1],
            gtr.rta * gtr.pi[2],
            gtr.rtg * gtr.pi[3],
            gtr.rtc * gtr.pi[0],
            -(gtr.rtc * gtr.pi[0] + gtr.rca * gtr.pi[2] + gtr.rcg * gtr.pi[3]),
            gtr.rca * gtr.pi[2],
            gtr.rcg * gtr.pi[3],
            gtr.rta * gtr.pi[0],
            gtr.rca * gtr.pi[1],
            -(gtr.rta * gtr.pi[0] + gtr.rca * gtr.pi[1] + gtr.rag * gtr.pi[3]),
            gtr.rag * gtr.pi[3],
            gtr.rtg * gtr.pi[0],
            gtr.rcg * gtr.pi[1],
            gtr.rag * gtr.pi[2],
            -(gtr.rtg * gtr.pi[0] + gtr.rcg * gtr.pi[1] + gtr.rag * gtr.pi[2]),
        ],
    )
}
