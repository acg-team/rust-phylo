use std::ops::Div;

use anyhow::bail;
use log::info;

use crate::substitution_models::{
    dna_models::{
        dna_substitution_parameters::DNASubstParams, make_dna_model, make_pi, DNASubstModel,
        FreqVector,
    },
    SubstMatrix,
};
use crate::Result;

pub fn tn93(tn93_params: DNASubstParams) -> DNASubstModel {
    info!(
        "Setting up tn93 with parameters {}",
        tn93_params.print_as_tn93()
    );
    let q = tn93_q(&tn93_params);
    make_dna_model(tn93_params, q)
}

pub fn parse_tn93_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
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
    let pi = make_pi(&[
        model_params[0],
        model_params[1],
        model_params[2],
        model_params[3],
    ])?;
    let alpha1 = model_params[4];
    let alpha2 = model_params[5];
    let beta = model_params[6];
    Ok(tn93_params(pi, alpha1, alpha2, beta))
}

fn tn93_params(pi: FreqVector, alpha1: f64, alpha2: f64, beta: f64) -> DNASubstParams {
    DNASubstParams {
        pi,
        rtc: alpha1,
        rta: beta,
        rtg: beta,
        rca: beta,
        rcg: beta,
        rag: alpha2,
    }
}

pub(crate) fn tn93_q(p: &DNASubstParams) -> SubstMatrix {
    let ft = p.pi[0];
    let fc = p.pi[1];
    let fa = p.pi[2];
    let fg = p.pi[3];
    let alpha1 = p.rtc;
    let alpha2 = p.rag;
    let beta = p.rta;
    let total = (alpha1 * fc + beta * (fa + fg)) * ft
        + (alpha1 * ft + beta * (fa + fg)) * fc
        + (beta * (ft + fc) + alpha2 * fg) * fa
        + (beta * (ft + fc) + alpha2 * fa) * fg;
    SubstMatrix::from_row_slice(
        4,
        4,
        &[
            -(alpha1 * fc + beta * (fa + fg)),
            alpha1 * fc,
            beta * fa,
            beta * fg,
            alpha1 * ft,
            -(alpha1 * ft + beta * (fa + fg)),
            beta * fa,
            beta * fg,
            beta * ft,
            beta * fc,
            -(beta * (ft + fc) + alpha2 * fg),
            alpha2 * fg,
            beta * ft,
            beta * fc,
            alpha2 * fa,
            -(beta * (ft + fc) + alpha2 * fa),
        ],
    )
    .div(total)
}
