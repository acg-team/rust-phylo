use std::ops::Div;

use anyhow::bail;

use crate::substitution_models::{
    dna_models::{make_dna_model, make_pi, DNASubstModel, FreqVector},
    SubstMatrix,
};
use crate::Result;

struct GtrParams<'a> {
    pi: &'a FreqVector,
    rtc: f64,
    rta: f64,
    rtg: f64,
    rca: f64,
    rcg: f64,
    rag: f64,
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
    let pi = make_pi(&[
        model_params[0],
        model_params[1],
        model_params[2],
        model_params[3],
    ])?;
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
