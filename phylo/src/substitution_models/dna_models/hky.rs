use anyhow::bail;
use log::{info, warn};

use crate::substitution_models::dna_models::{
    dna_substitution_parameters::DNASubstParams, make_dna_model, make_pi, tn93_q, DNASubstModel,
};
use crate::substitution_models::FreqVector;
use crate::Result;

pub fn hky(model_params: &[f64]) -> Result<DNASubstModel> {
    let hky_params = parse_hky_parameters(model_params)?;
    info!(
        "Setting up hky with parameters {}",
        hky_params.print_as_hky()
    );
    let q = tn93_q(&hky_params);
    Ok(make_dna_model(hky_params, q))
}

pub fn parse_hky_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
    if model_params.len() < 4 {
        bail!(
            "Too few parameters for the hky model, expected at least 4, got {}",
            model_params.len()
        );
    }
    let (alpha, beta) = if model_params.len() == 4 {
        warn!("Too few values provided for HKY, required pi and 1 or 2 values, kappa or alpha and beta.");
        warn!("Falling back to default values.");
        (2.0, 1.0)
    } else if model_params.len() == 5 {
        (model_params[4], 1.0)
    } else if model_params.len() == 6 {
        (model_params[4], model_params[5])
    } else {
        warn!("Too many values provided for HKY, required pi and 1 or 2 values, kappa or alpha and beta.");
        warn!("Will only use the first values provided.");
        (model_params[4], model_params[5])
    };
    let pi = make_pi(&[
        model_params[0],
        model_params[1],
        model_params[2],
        model_params[3],
    ])?;
    Ok(hky_params(pi, alpha, beta))
}

fn hky_params(pi: FreqVector, alpha: f64, beta: f64) -> DNASubstParams {
    DNASubstParams {
        pi,
        rtc: alpha,
        rta: beta,
        rtg: beta,
        rca: beta,
        rcg: beta,
        rag: alpha,
    }
}
