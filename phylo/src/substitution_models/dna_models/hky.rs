use anyhow::bail;
use log::info;

use crate::substitution_models::dna_models::{
    make_dna_model, make_pi, tn93_q, DNASubstModel, TN93Params,
};
use crate::Result;

pub fn hky(model_params: &[f64]) -> Result<DNASubstModel> {
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
    Ok(make_dna_model(
        model_params[0..5].to_vec(),
        tn93_q(hky_params),
        pi,
    ))
}
