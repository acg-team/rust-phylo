use anyhow::bail;

use crate::evolutionary_models::EvolutionaryModel;
use crate::substitution_models::{dna_models::DNASubstModel, SubstMatrix, SubstitutionModel};
use crate::Result;

#[derive(Clone, Debug)]
pub struct PIPModel<const N: usize> {
    pub subst_model: SubstitutionModel<N>,
    pub lambda: f64,
    pub mu: f64,
}

#[allow(dead_code)]
#[allow(unused_variables)]
impl EvolutionaryModel<4> for PIPModel<4> {
    fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
    where
        Self: std::marker::Sized,
    {
        if model_params.len() < 2 {
            bail!("Too few values provided for PIP, required 2 values, lambda and mu.");
        }
        let lambda = model_params[0];
        let mu = model_params[1];
        let subst_model = DNASubstModel::new(model_name, &model_params[2..])?;
        let model = PIPModel {
            subst_model,
            lambda,
            mu,
        };
        Ok(model)
    }

    fn get_p(&self, time: f64) -> SubstMatrix {
        debug_assert!(time >= 0.0);
        todo!()
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        todo!()
    }

    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounded: bool,
    ) -> std::collections::HashMap<ordered_float::OrderedFloat<f64>, (SubstMatrix, f64)> {
        todo!()
    }

    fn normalise(&mut self) {
        todo!()
    }

    fn get_scoring_matrix(&self, time: f64, rounded: bool) -> (SubstMatrix, f64) {
        todo!()
    }

    fn get_stationary_distribution(&self) -> &crate::substitution_models::FreqVector {
        todo!()
    }

    fn get_char_probability(&self, char: u8) -> nalgebra::DVector<f64> {
        todo!()
    }
}

// impl<const N: usize> EvolutionaryModel<N> for PIPModel<'_, N> {
//     fn new(model_name: &str, model_params: &[f64]) -> Result<Self>
//     where
//         Self: std::marker::Sized,
//     {
//         if model_params.len() < 2 {
//             bail!("Too few values provided for PIP, required 2 values, lambda and mu.");
//         }
//         let lambda = model_params[0];
//         let mu = model_params[1];
//         let subst_model = SubstitutionModel::<N>::new(model_name, model_params[2..])?;
//         let model = PIPModel {
//             subst_model,
//             lambda,
//             mu,
//         };
//         Ok(model)
//     }

//     fn get_p(&self, time: f64) -> crate::substitution_models::SubstMatrix<N> {
//         todo!()
//     }

//     fn get_rate(&self, i: u8, j: u8) -> f64 {
//         todo!()
//     }

//     fn generate_scorings(
//         &self,
//         times: &[f64],
//         zero_diag: bool,
//         rounded: bool,
//     ) -> std::collections::HashMap<
//         ordered_float::OrderedFloat<f64>,
//         (crate::substitution_models::SubstMatrix<N>, f64),
//     > {
//         todo!()
//     }

//     fn normalise(&mut self) {
//         todo!()
//     }

//     fn get_scoring_matrix(
//         &self,
//         time: f64,
//         rounded: bool,
//     ) -> (crate::substitution_models::SubstMatrix<N>, f64) {
//         todo!()
//     }

//     fn get_stationary_distribution(&self) -> &crate::substitution_models::FreqVector<N> {
//         todo!()
//     }

//     fn get_char_probability(&self, char: u8) -> nalgebra::DVector<f64> {
//         todo!()
//     }
// }
