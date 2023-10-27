use crate::sequences::NUCLEOTIDES_STR;
use crate::Rounding;
use crate::{f64_h, Result};
use anyhow::anyhow;
use bio::io::fasta::Record;
use nalgebra::{Const, DMatrix, DVector, DimMin};
use ordered_float::OrderedFloat;
use std::collections::HashMap;

pub mod dna_models;
pub mod protein_models;

pub type SubstMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[derive(Clone, Debug, PartialEq)]
pub struct SubstitutionModel<const N: usize> {
    index: [i32; 255],
    pub q: SubstMatrix,
    pub pi: FreqVector,
}

impl<const N: usize> SubstitutionModel<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn get_p(&self, time: f64) -> SubstMatrix {
        (self.q.clone() * time).exp()
    }

    fn get_rate(&self, i: u8, j: u8) -> f64 {
        assert!(
            self.index[i as usize] >= 0 && self.index[j as usize] >= 0,
            "Invalid rate requested."
        );
        self.q[(
            self.index[i as usize] as usize,
            self.index[j as usize] as usize,
        )]
    }

    fn generate_scorings(
        &self,
        times: &[f64],
        zero_diag: bool,
        rounding: &Rounding,
    ) -> HashMap<OrderedFloat<f64>, (SubstMatrix, f64)> {
        HashMap::<f64_h, (SubstMatrix, f64)>::from_iter(times.iter().map(|&time| {
            (
                f64_h::from(time),
                self.get_scoring_matrix_corrected(time, zero_diag, rounding),
            )
        }))
    }

    fn normalise(&mut self) {
        let factor = -(self.pi.transpose() * self.q.diagonal())[(0, 0)];
        self.q /= factor;
    }

    fn get_scoring_matrix(&self, time: f64, rounding: &Rounding) -> (SubstMatrix, f64) {
        self.get_scoring_matrix_corrected(time, false, rounding)
    }

    fn get_stationary_distribution(&self) -> &FreqVector {
        &self.pi
    }

    fn get_scoring_matrix_corrected(
        &self,
        time: f64,
        zero_diag: bool,
        rounding: &Rounding,
    ) -> (SubstMatrix, f64) {
        let p = self.get_p(time);
        let mut scores = p.map(|x| -x.ln());
        if rounding.round {
            scores = scores.map(|x| {
                (x * 10.0_f64.powf(rounding.digits as f64)).round()
                    / 10.0_f64.powf(rounding.digits as f64)
            });
        }
        if zero_diag {
            scores.fill_diagonal(0.0);
        }

        (scores, scores.mean())
    }
}

#[cfg(test)]
mod substitution_models_tests;
