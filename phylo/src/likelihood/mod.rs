pub trait LikelihoodCostFunction {
    type Model;
    type Info;
    fn logl(&self) -> f64;
}

#[cfg(test)]
mod likelihood_tests;
