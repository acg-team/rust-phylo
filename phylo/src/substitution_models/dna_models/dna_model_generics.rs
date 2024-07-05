use std::ops::Div;

use anyhow::bail;
use lazy_static::lazy_static;
use log::warn;

use crate::evolutionary_models::DNAModelType;
use crate::sequences::{GAP, NUCLEOTIDES};
use crate::substitution_models::{dna_models::DNASubstParams, FreqVector, SubstMatrix};
use crate::{frequencies, Result};

fn make_pi(pi_array: &[f64]) -> Result<FreqVector> {
    let pi = frequencies!(pi_array);
    debug_assert!(
        pi.len() == 4,
        "There have to be 4 equilibrium frequencies for DNA models."
    );
    if pi.sum() - 1.0 > f64::EPSILON {
        bail!("The equilibrium frequencies provided do not sum up to 1.");
    }
    Ok(pi)
}

fn make_q(q_array: &[f64]) -> SubstMatrix {
    SubstMatrix::from_row_slice(4, 4, q_array)
}

pub(crate) fn jc69_params(params: &[f64]) -> Result<DNASubstParams> {
    if !params.is_empty() {
        warn!("Too many values provided for JC69, average rate is fixed at 1.0.");
    }
    Ok(DNASubstParams {
        model_type: DNAModelType::JC69,
        pi: make_pi(&[0.25; 4])?,
        rtc: 1.0,
        rta: 1.0,
        rtg: 1.0,
        rca: 1.0,
        rcg: 1.0,
        rag: 1.0,
    })
}

pub(crate) fn jc69_q() -> SubstMatrix {
    let r = 1.0 / 3.0;
    make_q(&[-1.0, r, r, r, r, -1.0, r, r, r, r, -1.0, r, r, r, r, -1.0])
}

pub(crate) fn k80_params(params: &[f64]) -> Result<DNASubstParams> {
    let (a, b) = if params.is_empty() {
        warn!("Too few values provided for K80, required 1 or 2 values, kappa or alpha and beta.");
        warn!("Falling back to default values.");
        (2.0, 1.0)
    } else if params.len() == 1 {
        (params[0], 1.0)
    } else if params.len() == 2 {
        (params[0], params[1])
    } else {
        warn!("Too many values provided for K80, required 2 values, alpha and beta.");
        warn!("Will only use the first two values provided.");
        (params[0], params[1])
    };
    Ok(DNASubstParams {
        model_type: DNAModelType::K80,
        pi: make_pi(&[0.25; 4])?,
        rtc: a,
        rta: b,
        rtg: b,
        rca: b,
        rcg: b,
        rag: a,
    })
}

pub(crate) fn k80_q(p: &DNASubstParams) -> SubstMatrix {
    let a = p.rtc;
    let b = p.rta;
    let total = a + 2.0 * b;
    make_q(&[
        -total, a, b, b, a, -total, b, b, b, b, -total, a, b, b, a, -total,
    ])
    .div(total)
}

pub(crate) fn hky_params(params: &[f64]) -> Result<DNASubstParams> {
    if params.len() < 4 {
        bail!(
            "Too few parameters for the hky model, expected at least 4, got {}",
            params.len()
        );
    }
    let (a, b) = if params.len() == 4 {
        warn!("Too few values provided for HKY, required pi and 1 or 2 values, kappa or alpha and beta.");
        warn!("Falling back to default values.");
        (2.0, 1.0)
    } else if params.len() == 5 {
        (params[4], 1.0)
    } else if params.len() == 6 {
        (params[4], params[5])
    } else {
        warn!("Too many values provided for HKY, required pi and 1 or 2 values, kappa or alpha and beta.");
        warn!("Will only use the first values provided.");
        (params[4], params[5])
    };
    Ok(DNASubstParams {
        model_type: DNAModelType::HKY,
        pi: make_pi(&[params[0], params[1], params[2], params[3]])?,
        rtc: a,
        rta: b,
        rtg: b,
        rca: b,
        rcg: b,
        rag: a,
    })
}

pub fn tn93_params(params: &[f64]) -> Result<DNASubstParams> {
    if params.len() != 7 {
        bail!(
            "{} parameters for the tn93 model, expected 7, got {}",
            if params.len() < 7 {
                "Not enough"
            } else {
                "Too many"
            },
            params.len()
        );
    }
    let a1 = params[4];
    let a2 = params[5];
    let b = params[6];
    Ok(DNASubstParams {
        model_type: DNAModelType::TN93,
        pi: make_pi(&[params[0], params[1], params[2], params[3]])?,
        rtc: a1,
        rta: b,
        rtg: b,
        rca: b,
        rcg: b,
        rag: a2,
    })
}

pub(crate) fn tn93_q(p: &DNASubstParams) -> SubstMatrix {
    let ft = p.pi[0];
    let fc = p.pi[1];
    let fa = p.pi[2];
    let fg = p.pi[3];
    let a1 = p.rtc;
    let a2 = p.rag;
    let b = p.rta;
    let total = (a1 * fc + b * (fa + fg)) * ft
        + (a1 * ft + b * (fa + fg)) * fc
        + (b * (ft + fc) + a2 * fg) * fa
        + (b * (ft + fc) + a2 * fa) * fg;
    make_q(&[
        -(a1 * fc + b * (fa + fg)),
        a1 * fc,
        b * fa,
        b * fg,
        a1 * ft,
        -(a1 * ft + b * (fa + fg)),
        b * fa,
        b * fg,
        b * ft,
        b * fc,
        -(b * (ft + fc) + a2 * fg),
        a2 * fg,
        b * ft,
        b * fc,
        a2 * fa,
        -(b * (ft + fc) + a2 * fa),
    ])
    .div(total)
}

pub fn gtr_params(params: &[f64]) -> Result<DNASubstParams> {
    if params.len() != 10 {
        bail!(
            "{} parameters for the GTR model, expected 10, got {}",
            if params.len() < 10 {
                "Not enough"
            } else {
                "Too many"
            },
            params.len()
        );
    }

    let gtr_params = DNASubstParams {
        model_type: DNAModelType::GTR,
        pi: make_pi(&[params[0], params[1], params[2], params[3]])?,
        rtc: params[4],
        rta: params[5],
        rtg: params[6],
        rca: params[7],
        rcg: params[8],
        rag: params[9],
    };

    Ok(gtr_params)
}

pub(crate) fn gtr_q(gtr: &DNASubstParams) -> SubstMatrix {
    let ft = gtr.pi[0];
    let fc = gtr.pi[1];
    let fa = gtr.pi[2];
    let fg = gtr.pi[3];
    let total = (gtr.rtc * fc + gtr.rta * fa + gtr.rtg * fg) * ft
        + (gtr.rtc * ft + gtr.rca * fa + gtr.rcg * fg) * fc
        + (gtr.rta * ft + gtr.rca * fc + gtr.rag * fg) * fa
        + (gtr.rtg * ft + gtr.rcg * fc + gtr.rag * fa) * fg;
    make_q(&[
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
    ])
    .div(total)
}

lazy_static! {
    pub static ref NUCLEOTIDE_INDEX: [usize; 255] = {
        let mut index = [0; 255];
        for (i, char) in NUCLEOTIDES.iter().enumerate() {
            index[*char as usize] = i;
            index[(*char).to_ascii_lowercase() as usize] = i;
        }
        index[GAP as usize] = 4;
        index
    };
    pub static ref DNA_GAP_SETS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 5]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            if char == GAP {
                elem.set_column(0, &frequencies!(&[0.0, 0.0, 0.0, 0.0, 1.0]));
            } else {
                elem.set_column(0, &generic_dna_sets(char).resize_vertically(5, 0.0));
            }
        }
        map
    };
    pub static ref DNA_SETS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &generic_dna_sets(char));
        }
        map
    };
}
fn generic_dna_sets(char: u8) -> FreqVector {
    match char {
        b'T' | b't' => frequencies!(&[1.0, 0.0, 0.0, 0.0]),
        b'C' | b'c' => frequencies!(&[0.0, 1.0, 0.0, 0.0]),
        b'A' | b'a' => frequencies!(&[0.0, 0.0, 1.0, 0.0]),
        b'G' | b'g' => frequencies!(&[0.0, 0.0, 0.0, 1.0]),
        b'M' | b'm' => frequencies!(&[0.0, 1.0 / 2.0, 1.0 / 2.0, 0.0]),
        b'R' | b'r' => frequencies!(&[0.0, 0.0, 1.0 / 2.0, 1.0 / 2.0]),
        b'W' | b'w' => frequencies!(&[1.0 / 2.0, 0.0, 1.0 / 2.0, 0.0]),
        b'S' | b's' => frequencies!(&[0.0, 1.0 / 2.0, 0.0, 1.0 / 2.0]),
        b'Y' | b'y' => frequencies!(&[1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0]),
        b'K' | b'k' => frequencies!(&[1.0 / 2.0, 0.0, 0.0, 1.0 / 2.0]),
        b'V' | b'v' => {
            frequencies!(&[0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        }
        b'D' | b'd' => {
            frequencies!(&[1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0])
        }
        b'B' | b'b' => {
            frequencies!(&[1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0])
        }
        b'H' | b'h' => {
            frequencies!(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0])
        }
        _ => frequencies!(&[1.0 / 4.0; 4]),
    }
}
