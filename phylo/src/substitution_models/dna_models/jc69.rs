use log::{info, warn};

use crate::substitution_models::{
    dna_models::{make_dna_model, DNASubstModel, DNASubstParams},
    FreqVector, SubstMatrix,
};
use crate::Result;

pub fn jc69(model_params: &[f64]) -> Result<DNASubstModel> {
    let jc69_params = parse_jc69_parameters(model_params)?;
    info!(
        "Setting up jc69 with parameters: {}",
        jc69_params.print_as_jc69()
    );
    Ok(make_dna_model(jc69_params, jc69_q()))
}

pub fn parse_jc69_parameters(model_params: &[f64]) -> Result<DNASubstParams> {
    if !model_params.is_empty() {
        warn!("Too many values provided for JC69, average rate is fixed at 1.0.");
    }
    Ok(DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.25; 4]),
        rtc: 1.0,
        rta: 1.0,
        rtg: 1.0,
        rca: 1.0,
        rcg: 1.0,
        rag: 1.0,
    })
}

pub fn jc69_q() -> SubstMatrix {
    let r = 1.0 / 3.0;
    SubstMatrix::from_row_slice(
        4,
        4,
        &[-1.0, r, r, r, r, -1.0, r, r, r, r, -1.0, r, r, r, r, -1.0],
    )
}
