use crate::phylo_info::PhyloInfo;

pub trait LikelihoodCostFunction {
    fn logl(&self, info: &PhyloInfo) -> f64;
}

#[cfg(test)]
mod tests;
