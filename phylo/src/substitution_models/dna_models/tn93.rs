use std::ops::Div;

use anyhow::bail;
use log::info;

use crate::substitution_models::{
    dna_models::{make_dna_model, make_pi, DNASubstModel, FreqVector},
    SubstMatrix,
};
use crate::Result;

pub(crate) struct TN93Params<'a> {
    pub(crate) pi: &'a FreqVector,
    pub(crate) a1: f64,
    pub(crate) a2: f64,
    pub(crate) b: f64,
}

pub fn tn93(model_params: &[f64]) -> Result<DNASubstModel> {
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
    Ok(make_dna_model(
        model_params[0..7].to_vec(),
        tn93_q(tn93_params),
        pi,
    ))
}

pub(crate) fn tn93_q(p: &TN93Params) -> SubstMatrix {
    let ft = p.pi[0];
    let fc = p.pi[1];
    let fa = p.pi[2];
    let fg = p.pi[3];
    let total = (p.a1 * fc + p.b * (fa + fg)) * ft
        + (p.a1 * ft + p.b * (fa + fg)) * fc
        + (p.b * (ft + fc) + p.a2 * fg) * fa
        + (p.b * (ft + fc) + p.a2 * fa) * fg;
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
    .div(total)
}
