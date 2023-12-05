use crate::substitution_models::FreqVector;

pub trait LikelihoodCostFunction<'a, const N: usize> {
    type Model;
    type Info;
    fn compute_log_likelihood(&self, model: &Self::Model, tmp_info: &mut Self::Info) -> f64;
    fn get_empirical_frequencies(&self) -> FreqVector;
}

#[cfg(test)]
mod likelihood_tests;
