use crate::alphabets::Alphabet;

#[derive(Clone, Debug, PartialEq)]
pub struct GapMultipliers {
    pub(crate) open: f64,
    pub(crate) ext: f64,
}

pub trait ParsimonyCosts {
    fn alphabet(&self) -> &Alphabet;
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64;
    fn gap_open(&self, blen: f64) -> f64;
    fn gap_ext(&self, blen: f64) -> f64;
    fn avg(&self, blen: f64) -> f64;
}

pub mod parsimony_costs_model;
pub mod parsimony_costs_simple;
pub(crate) use parsimony_costs_simple::*;

#[cfg(test)]
mod tests;
