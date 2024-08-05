use crate::substitution_models::FreqVector;

pub trait LikelihoodCostFunction {
    type Model;
    type Info;
    fn compute_logl(&self) -> f64;
    fn empirical_frequencies(&self) -> FreqVector;
}

#[cfg(test)]
mod likelihood_tests;
