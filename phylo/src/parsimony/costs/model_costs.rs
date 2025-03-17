use log::{debug, info};
use ordered_float::OrderedFloat;

use crate::alphabets::Alphabet;
use crate::parsimony::{CostMatrix, DiagonalZeros, GapCost, ParsimonyCosts, ParsimonyModel, Rounding};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelCosts {
    alphabet: Alphabet,
    costs: Vec<(OrderedFloat<f64>, ModelBranchCosts)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelBranchCosts {
    avg: f64,
    gap: GapCost,
    c: CostMatrix,
}

impl ModelCosts {
    pub fn new(
        model: &dyn ParsimonyModel,
        gap: GapCost,
        diagonal: DiagonalZeros,
        rounding: Rounding,
        times: &[f64],
    ) -> Result<Self> {
        info!(
            "Setting up the parsimony scoring from the {} substitution model.",
            model
        );

        let mut times = times
            .iter()
            .cloned()
            .map(OrderedFloat::<f64>::from)
            .collect::<Vec<_>>();
        times.sort();

        let costs: Vec<(OrderedFloat<f64>, ModelBranchCosts)> = times
            .iter()
            .map(|time| {
                let (cost_matrix, avg) =
                    model.scoring_corrected(f64::from(*time), diagonal.clone(), rounding.clone());
                (
                    *time,
                    ModelBranchCosts {
                        avg,
                        gap: gap.clone() * avg,
                        c: cost_matrix,
                    },
                )
            })
            .collect();

        info!(
            "Created scoring matrices from the {} substitution model for {:?} branch lengths.",
            model, times
        );
        debug!("The scoring matrices are: {:?}", costs);
        Ok(ModelCosts {
            alphabet: model.alphabet().clone(),
            costs,
        })
    }
}

impl ModelCosts {
    fn scoring(&self, target: OrderedFloat<f64>) -> &ModelBranchCosts {
        let target = OrderedFloat(target);
        match self.costs.binary_search_by(|(time, _)| time.cmp(&target)) {
            Ok(idx) => &self.costs[idx].1, // Exact match
            Err(idx) => {
                if idx == 0 {
                    &self.costs[0].1
                } else if idx == self.costs.len() {
                    &self.costs[self.costs.len() - 1].1
                } else {
                    // Pick closest value
                    let (before, _) = self.costs[idx - 1];
                    let (after, _) = self.costs[idx];
                    if (target.0 - before.0).abs() <= (after - target.0).abs() {
                        &self.costs[idx - 1].1
                    } else {
                        &self.costs[idx].1
                    }
                }
            }
        }
    }
}

impl ParsimonyCosts for ModelCosts {
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64 {
        self.scoring(OrderedFloat(blen)).c[(self.alphabet.index(i), self.alphabet.index(j))]
    }

    fn gap_open(&self, blen: f64) -> f64 {
        self.scoring(OrderedFloat(blen)).gap.open
    }

    fn gap_ext(&self, blen: f64) -> f64 {
        self.scoring(OrderedFloat(blen)).gap.ext
    }

    fn avg(&self, blen: f64) -> f64 {
        self.scoring(OrderedFloat(blen)).avg
    }
}

#[cfg(test)]
mod private_tests {
    use approx::assert_relative_eq;

    use crate::substitution_models::{SubstModel, HIVB, K80, WAG};

    use super::*;
    use super::{DiagonalZeros as Z, Rounding as R};

    const TRUE_COST_MATRIX: [f64; 400] = [
        0.0, 6.0, 6.0, 5.0, 6.0, 6.0, 5.0, 4.0, 7.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 4.0, 4.0, 9.0,
        7.0, 4.0, 5.0, 0.0, 6.0, 7.0, 7.0, 5.0, 6.0, 5.0, 5.0, 7.0, 5.0, 3.0, 7.0, 8.0, 6.0, 5.0,
        6.0, 6.0, 7.0, 6.0, 5.0, 6.0, 0.0, 4.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 4.0, 8.0, 8.0,
        7.0, 4.0, 4.0, 9.0, 6.0, 6.0, 5.0, 7.0, 4.0, 0.0, 9.0, 6.0, 3.0, 5.0, 6.0, 8.0, 7.0, 6.0,
        8.0, 8.0, 6.0, 5.0, 6.0, 9.0, 7.0, 7.0, 5.0, 6.0, 7.0, 8.0, 0.0, 8.0, 8.0, 6.0, 7.0, 7.0,
        6.0, 7.0, 7.0, 6.0, 7.0, 5.0, 6.0, 7.0, 6.0, 5.0, 5.0, 4.0, 5.0, 6.0, 8.0, 0.0, 4.0, 6.0,
        5.0, 7.0, 5.0, 4.0, 6.0, 8.0, 5.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 6.0, 6.0, 3.0, 9.0, 4.0,
        0.0, 5.0, 6.0, 7.0, 7.0, 4.0, 7.0, 8.0, 6.0, 5.0, 5.0, 8.0, 7.0, 5.0, 4.0, 6.0, 5.0, 5.0,
        7.0, 7.0, 6.0, 0.0, 7.0, 8.0, 7.0, 6.0, 8.0, 8.0, 7.0, 5.0, 6.0, 8.0, 8.0, 7.0, 6.0, 5.0,
        4.0, 5.0, 8.0, 4.0, 6.0, 6.0, 0.0, 7.0, 5.0, 5.0, 7.0, 6.0, 6.0, 5.0, 6.0, 8.0, 4.0, 7.0,
        6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 0.0, 4.0, 6.0, 5.0, 5.0, 8.0, 6.0, 5.0, 8.0,
        6.0, 3.0, 6.0, 6.0, 7.0, 8.0, 7.0, 6.0, 7.0, 7.0, 7.0, 4.0, 0.0, 6.0, 5.0, 5.0, 6.0, 6.0,
        6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 4.0, 6.0, 9.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 0.0, 6.0, 8.0,
        6.0, 5.0, 5.0, 8.0, 8.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 4.0, 3.0, 5.0,
        0.0, 5.0, 7.0, 6.0, 5.0, 7.0, 6.0, 4.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 8.0, 6.0, 5.0,
        4.0, 7.0, 6.0, 0.0, 7.0, 6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 6.0, 7.0, 6.0, 8.0, 6.0, 6.0, 6.0,
        6.0, 7.0, 6.0, 6.0, 8.0, 7.0, 0.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 5.0, 4.0, 5.0, 6.0, 6.0,
        5.0, 5.0, 6.0, 6.0, 6.0, 5.0, 7.0, 6.0, 5.0, 0.0, 4.0, 7.0, 6.0, 6.0, 4.0, 6.0, 5.0, 6.0,
        7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 4.0, 0.0, 9.0, 7.0, 5.0, 7.0, 5.0,
        8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 6.0, 7.0, 0.0, 5.0, 6.0,
        6.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 7.0, 4.0, 7.0, 5.0, 6.0, 6.0,
        0.0, 6.0, 4.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        5.0, 8.0, 7.0, 0.0,
    ];

    #[test]
    fn generate_protein_scorings() {
        let s = ModelCosts::new(
            &SubstModel::<WAG>::new(&[], &[]),
            GapCost::new(2.5, 1.0),
            Z::non_zero(),
            R::zero(),
            &[0.1, 0.3, 0.5, 0.7],
        )
        .unwrap();

        let mat_01 = s.scoring(OrderedFloat(0.1));
        let true_matrix_01 = CostMatrix::from_row_slice(20, 20, &TRUE_COST_MATRIX);
        assert_relative_eq!(mat_01.c, true_matrix_01);
        assert_relative_eq!(mat_01.avg, 5.7675);

        assert_relative_eq!(s.avg(0.3), 4.7475);
        assert_relative_eq!(s.avg(0.5), 4.2825);
        assert_relative_eq!(s.avg(0.7), 4.0075);
    }

    #[test]
    fn protein_scoring_matrices() {
        let model = SubstModel::<WAG>::new(&[], &[]);
        let true_matrix_01 = CostMatrix::from_row_slice(20, 20, &TRUE_COST_MATRIX);
        let (mat, avg) = model.scoring(0.1, R::zero());

        assert_relative_eq!(mat, true_matrix_01);

        assert_relative_eq!(avg, 5.7675);
        let (_, avg) = model.scoring(0.3, R::zero());
        assert_relative_eq!(avg, 4.7475);
        let (_, avg) = model.scoring(0.5, R::zero());
        assert_relative_eq!(avg, 4.2825);
        let (_, avg) = model.scoring(0.7, R::zero());
        assert_relative_eq!(avg, 4.0075);
    }

    #[test]
    fn matrix_entry_rounding() {
        let model = SubstModel::<K80>::new(&[], &[1.0, 2.0]);
        let (mat_round, avg_round) = model.scoring_corrected(0.1, Z::zero(), R::zero());
        let (mat, avg) = model.scoring_corrected(0.1, Z::zero(), R::none());
        assert_ne!(avg_round, avg);
        assert_ne!(mat_round, mat);
        for &element in mat_round.as_slice() {
            assert_eq!(element.round(), element);
        }
        let model = SubstModel::<HIVB>::new(&[], &[]);
        let (mat_round, avg_round) = model.scoring_corrected(0.1, Z::zero(), R::zero());
        let (mat, avg) = model.scoring_corrected(0.1, Z::zero(), R::none());
        assert_ne!(avg_round, avg);
        assert_ne!(mat_round, mat);
        for &element in mat_round.as_slice() {
            assert_eq!(element.round(), element);
        }
    }

    #[test]
    fn matrix_zero_diagonals() {
        let model = SubstModel::<HIVB>::new(&[], &[]);
        let (mat_zeros, avg_zeros) = model.scoring_corrected(0.5, Z::zero(), R::zero());
        let (mat, avg) = model.scoring_corrected(0.5, Z::non_zero(), R::zero());
        assert_ne!(avg_zeros, avg);
        assert!(avg_zeros < avg);
        assert_ne!(mat_zeros, mat);
        for &element in mat_zeros.diagonal().iter() {
            assert_eq!(element, 0.0);
        }
    }
}
