use std::ops::Div;

use anyhow::bail;
use log::info;

use crate::substitution_models::{
    dna_models::{make_dna_model, make_pi, DNASubstModel, DNASubstParams},
    SubstMatrix,
};
use crate::Result;

pub fn gtr(gtr_params: DNASubstParams) -> DNASubstModel {
    info!("Setting up GTR with rates: {}", gtr_params.print_as_gtr());
    let q = gtr_q(&gtr_params);
    make_dna_model(gtr_params, q)
}

pub fn parse_gtr_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
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
    let gtr_params = DNASubstParams {
        pi,
        rtc: model_params[4],
        rta: model_params[5],
        rtg: model_params[6],
        rca: model_params[7],
        rcg: model_params[8],
        rag: model_params[9],
    };

    Ok(gtr_params)
}

fn gtr_q(gtr: &DNASubstParams) -> SubstMatrix {
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
