#[derive(Clone, Debug, PartialEq)]
pub struct GapMultipliers {
    pub(crate) open: f64,
    pub(crate) ext: f64,
}

pub trait ParsimonyCosts {
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64;
    fn gap_open(&self, blen: f64) -> f64;
    fn gap_ext(&self, blen: f64) -> f64;
    fn avg(&self, blen: f64) -> f64;
}

pub mod model_costs;
#[allow(unused_imports)]
pub(crate) use model_costs::*;
pub mod simple_costs;
pub(crate) use simple_costs::*;

#[cfg(test)]
mod tests;
