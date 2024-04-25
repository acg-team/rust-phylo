use std::ops::Div;

use log::{info, warn};

use crate::substitution_models::{
    dna_models::{dna_substitution_parameters::DNASubstParams, make_dna_model, DNASubstModel},
    FreqVector, SubstMatrix,
};
use crate::Result;

pub fn k80(k80_params: DNASubstParams) -> DNASubstModel {
    info!(
        "Setting up k80 with parameters: {}",
        k80_params.print_as_k80()
    );
    let q = k80_q(&k80_params);
    make_dna_model(k80_params, q)
}

pub fn parse_k80_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
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
    Ok(k80_params(alpha, beta))
}

fn k80_params(alpha: f64, beta: f64) -> DNASubstParams {
    DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.25; 4]),
        rtc: alpha,
        rta: beta,
        rtg: beta,
        rca: beta,
        rcg: beta,
        rag: alpha,
    }
}

pub fn k80_q(p: &DNASubstParams) -> SubstMatrix {
    let alpha = p.rtc;
    let beta = p.rta;
    let total = alpha + 2.0 * beta;
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
    .div(total)
}
