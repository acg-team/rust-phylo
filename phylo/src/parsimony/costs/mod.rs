use std::ops::Mul;

#[derive(Clone, Debug, PartialEq)]
pub struct GapCost {
    open: f64,
    ext: f64,
}

impl Mul<f64> for GapCost {
    type Output = GapCost;

    fn mul(self, rhs: f64) -> GapCost {
        GapCost {
            open: self.open * rhs,
            ext: self.ext * rhs,
        }
    }
}

impl GapCost {
    pub fn new(open: f64, ext: f64) -> GapCost {
        GapCost { open, ext }
    }
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
