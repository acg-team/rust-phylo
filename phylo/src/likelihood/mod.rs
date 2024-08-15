pub trait LikelihoodCostFunction {
    type Model;
    type Info;
    fn compute_logl(&self) -> f64;
}

#[cfg(test)]
mod likelihood_tests;
