use crate::phylo_info::PhyloInfo;

pub trait PhyloCostFunction {
    fn cost(&self, info: &PhyloInfo, reset: bool) -> f64;
    fn reset(&self);
}

#[cfg(test)]
mod tests;
