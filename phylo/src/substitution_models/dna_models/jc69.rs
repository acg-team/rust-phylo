use log::warn;

use crate::substitution_models::{
    dna_models::{make_dna_model, DNASubstModel},
    FreqVector, SubstMatrix,
};
use crate::Result;

pub fn jc69(model_params: &[f64]) -> Result<DNASubstModel> {
    if !model_params.is_empty() {
        warn!("Too many values provided for JC69.");
    }
    Ok(make_dna_model(
        vec![],
        jc69_q(),
        FreqVector::from_column_slice(&[0.25; 4]),
    ))
}

pub fn jc69_q() -> SubstMatrix {
    let r = 1.0 / 3.0;
    SubstMatrix::from_row_slice(
        4,
        4,
        &[-1.0, r, r, r, r, -1.0, r, r, r, r, -1.0, r, r, r, r, -1.0],
    )
}
