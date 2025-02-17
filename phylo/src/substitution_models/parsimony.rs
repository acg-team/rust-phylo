use std::collections::HashMap;

use ordered_float::OrderedFloat;

use crate::evolutionary_models::EvoModel;
use crate::f64_h;
use crate::substitution_models::{QMatrix, SubstMatrix, SubstModel, SubstitutionModel};
use crate::Rounding;

pub trait ParsimonyModel {
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)>;

    fn scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64);

    fn scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64);
}

impl<Q: QMatrix> ParsimonyModel for SubstModel<Q>
where
    SubstModel<Q>: SubstitutionModel,
{
    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        HashMap::<f64_h, (SubstMatrix, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.scoring_matrix_corrected(time, zero_diag, rounding),
            )
        }))
    }

    fn scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.scoring_matrix_corrected(time, false, rounding)
    }

    fn scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64) {
        let mut scores = self.p(time);
        scores.iter_mut().for_each(|x| *x = -(*x).ln());
        if rounding.round {
            scores.iter_mut().for_each(|x| {
                *x = (*x * 10.0_f64.powf(rounding.digits as f64)).round()
                    / 10.0_f64.powf(rounding.digits as f64)
            });
        }
        if zero_diag {
            scores.fill_diagonal(0.0);
        }
        let mean = scores.mean();
        (scores, mean)
    }
}
