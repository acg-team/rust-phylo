use crate::alphabets::Alphabet;

pub trait ParsimonyCosts {
    fn branch_costs(&self, branch_length: f64) -> &dyn BranchParsimonyCosts;
    fn alphabet(&self) -> &Alphabet;
}

pub trait BranchParsimonyCosts {
    fn r#match(&self, i: u8, j: u8) -> f64;
    fn gap_open(&self) -> f64;
    fn gap_ext(&self) -> f64;
    fn avg(&self) -> f64;
}

// pub mod parsimony_costs_model;

pub mod parsimony_costs_simple;
pub(crate) use parsimony_costs_simple::*;

#[cfg(test)]
mod tests;
