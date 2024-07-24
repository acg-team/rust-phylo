use crate::substitution_models::FreqVector;

pub trait LikelihoodCostFunction {
    type Model;
    type Info;
    fn compute_log_likelihood(&self) -> f64;
    fn get_empirical_frequencies(&self) -> FreqVector;
}

#[cfg(test)]
mod likelihood_tests;
