use std::collections::HashMap;

use log::{debug, info};

use crate::alphabets::Alphabet;
use crate::parsimony::{CostMatrix, GapCost, ParsimonyCosts, ParsimonyModel, Rounding, Zeroing};
use crate::{cmp_f64, ord_f64, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct ModelCosts {
    alphabet: Alphabet,
    times: Vec<f64>,
    costs: HashMap<ord_f64, ModelBranchCosts>,
}

impl ModelCosts {
    pub fn new(
        model: &dyn ParsimonyModel,
        gap: GapCost,
        diagonal_zeros: Zeroing,
        rounding: Rounding,
        times: &[f64],
    ) -> Result<Self> {
        info!(
            "Setting up the parsimony scoring from the {} substitution model.",
            model
        );

        let mut costs = HashMap::<ord_f64, ModelBranchCosts>::with_capacity(times.len());

        for time in times {
            let (cost_matrix, avg) =
                model.scoring_matrix_corrected(*time, diagonal_zeros.clone(), rounding.clone());
            costs.insert(
                ord_f64::from(*time),
                ModelBranchCosts {
                    avg,
                    gap: gap.clone() * avg,
                    c: cost_matrix,
                },
            );
        }

        info!(
            "Created scoring matrices from the {} substitution model for {:?} branch lengths.",
            model, times
        );
        debug!("The scoring matrices are: {:?}", costs);
        Ok(ModelCosts {
            alphabet: model.alphabet().clone(),
            times: sort_times(times),
            costs,
        })
    }
}

fn sort_times(times: &[f64]) -> Vec<f64> {
    let mut sorted_times = Vec::from(times);
    sorted_times.sort_by(cmp_f64());
    sorted_times
}

impl ModelCosts {
    fn find_closest_branch_length(&self, target: f64) -> f64 {
        debug!("Getting scoring for time {}", target);
        let time = match self
            .times
            .windows(2)
            .filter(|&window| target - window[0] > window[1] - target)
            .last()
        {
            Some(window) => window[1],
            None => self.times[0],
        };
        debug!("Using scoring for time {}", time);
        time
    }
}

impl ParsimonyCosts for ModelCosts {
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64 {
        self.costs[&ord_f64::from(self.find_closest_branch_length(blen))].c
            [(self.alphabet.index(i), self.alphabet.index(j))]
    }

    fn gap_open(&self, blen: f64) -> f64 {
        self.costs[&ord_f64::from(self.find_closest_branch_length(blen))]
            .gap
            .open
    }

    fn gap_ext(&self, blen: f64) -> f64 {
        self.costs[&ord_f64::from(self.find_closest_branch_length(blen))]
            .gap
            .ext
    }

    fn avg(&self, blen: f64) -> f64 {
        self.costs
            .get(&ord_f64::from(self.find_closest_branch_length(blen)))
            .unwrap()
            .avg
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ModelBranchCosts {
    avg: f64,
    gap: GapCost,
    c: CostMatrix,
}

#[cfg(test)]
mod private_tests {
    use approx::assert_relative_eq;

    use crate::parsimony::costs::ParsimonyCosts;
    use crate::parsimony::{CostMatrix, ParsimonyModel, Rounding as R};
    use crate::substitution_models::{SubstModel, HIVB, K80, WAG};

    use super::Zeroing as Z;
    use super::*;

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
        let model = SubstModel::<WAG>::new(&[], &[]);
        let scoring = ModelCosts::new(
            &model,
            GapCost::new(2.5, 1.0),
            Z::no(),
            R::zero(),
            &[0.1, 0.3, 0.5, 0.7],
        )
        .unwrap();

        let mat_01 = scoring
            .costs
            .get(&ordered_float::OrderedFloat(0.1))
            .unwrap();
        let true_matrix_01 = CostMatrix::from_row_slice(20, 20, &TRUE_COST_MATRIX);
        assert_relative_eq!(mat_01.c, true_matrix_01);
        assert_relative_eq!(mat_01.avg, 5.7675);

        assert_relative_eq!(scoring.avg(0.3), 4.7475);
        assert_relative_eq!(scoring.avg(0.5), 4.2825);
        assert_relative_eq!(scoring.avg(0.7), 4.0075);
    }

    #[test]
    fn protein_scoring_matrices() {
        let model = SubstModel::<WAG>::new(&[], &[]);
        let true_matrix_01 = CostMatrix::from_row_slice(20, 20, &TRUE_COST_MATRIX);
        let (mat, avg) = ParsimonyModel::scoring_matrix(&model, 0.1, R::zero());

        assert_relative_eq!(mat, true_matrix_01);

        assert_relative_eq!(avg, 5.7675);
        let (_, avg) = ParsimonyModel::scoring_matrix(&model, 0.3, R::zero());
        assert_relative_eq!(avg, 4.7475);
        let (_, avg) = ParsimonyModel::scoring_matrix(&model, 0.5, R::zero());
        assert_relative_eq!(avg, 4.2825);
        let (_, avg) = ParsimonyModel::scoring_matrix(&model, 0.7, R::zero());
        assert_relative_eq!(avg, 4.0075);
    }

    #[test]
    fn matrix_entry_rounding() {
        let model = SubstModel::<K80>::new(&[], &[1.0, 2.0]);
        let (mat_round, avg_round) = model.scoring_matrix_corrected(0.1, Z::yes(), R::zero());
        let (mat, avg) = model.scoring_matrix_corrected(0.1, Z::yes(), R::none());
        assert_ne!(avg_round, avg);
        assert_ne!(mat_round, mat);
        for &element in mat_round.as_slice() {
            assert_eq!(element.round(), element);
        }
        let model = SubstModel::<HIVB>::new(&[], &[]);
        let (mat_round, avg_round) = model.scoring_matrix_corrected(0.1, Z::yes(), R::zero());
        let (mat, avg) = model.scoring_matrix_corrected(0.1, Z::yes(), R::none());
        assert_ne!(avg_round, avg);
        assert_ne!(mat_round, mat);
        for &element in mat_round.as_slice() {
            assert_eq!(element.round(), element);
        }
    }

    #[test]
    fn matrix_zero_diagonals() {
        let model = SubstModel::<HIVB>::new(&[], &[]);
        let (mat_zeros, avg_zeros) = model.scoring_matrix_corrected(0.5, Z::yes(), R::zero());
        let (mat, avg) = model.scoring_matrix_corrected(0.5, Z::no(), R::zero());
        assert_ne!(avg_zeros, avg);
        assert!(avg_zeros < avg);
        assert_ne!(mat_zeros, mat);
        for &element in mat_zeros.diagonal().iter() {
            assert_eq!(element, 0.0);
        }
    }
}
