use log::{debug, info};
use ordered_float::OrderedFloat;

use crate::alphabets::{Alphabet, ParsimonySet};
use crate::parsimony::{
    CostMatrix, DiagonalZeros, GapCost, ParsimonyModel, ParsimonyScoring, Rounding,
};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelScoringBuilder<P: ParsimonyModel> {
    model: P,
    gap: GapCost,
    diagonal: DiagonalZeros,
    rounding: Rounding,
    times: Vec<f64>,
}

impl<P: ParsimonyModel> ModelScoringBuilder<P> {
    pub fn new(model: P) -> Self {
        ModelScoringBuilder {
            model,
            gap: GapCost::new(2.5, 1.0),
            diagonal: DiagonalZeros::non_zero(),
            rounding: Rounding::none(),
            times: Vec::new(),
        }
    }

    pub fn gap_cost(mut self, gap: GapCost) -> Self {
        self.gap = gap;
        self
    }

    pub fn diagonal(mut self, diagonal: DiagonalZeros) -> Self {
        self.diagonal = diagonal;
        self
    }

    pub fn rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    pub fn times(mut self, times: Vec<f64>) -> Self {
        self.times = times;
        self
    }

    pub fn build(self) -> Result<ModelScoring> {
        info!(
            "Setting up the parsimony scoring from the {} model.",
            self.model
        );

        let mut times = self
            .times
            .iter()
            .cloned()
            .map(OrderedFloat::<f64>::from)
            .collect::<Vec<_>>();
        times.sort();

        let costs: Vec<(OrderedFloat<f64>, TimeCosts)> = times
            .iter()
            .map(|time| {
                let cost_matrix =
                    self.model
                        .scoring(f64::from(*time), &self.diagonal, &self.rounding);
                let avg = cost_matrix.mean();
                (
                    *time,
                    TimeCosts {
                        avg,
                        gap: self.gap * avg,
                        c: cost_matrix,
                    },
                )
            })
            .collect();

        info!(
            "Created scoring matrices from the {} substitution model for {:?} branch lengths.",
            self.model, times
        );
        debug!("The scoring matrices are: {:?}", costs);
        Ok(ModelScoring {
            alphabet: *self.model.alphabet(),
            costs,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelScoring {
    alphabet: Alphabet,
    costs: Vec<(OrderedFloat<f64>, TimeCosts)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimeCosts {
    avg: f64,
    gap: GapCost,
    c: CostMatrix,
}

impl ModelScoring {
    fn scoring(&self, target: OrderedFloat<f64>) -> &TimeCosts {
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
                    if (target.0 - before).abs() <= (after - target.0).abs() {
                        &self.costs[idx - 1].1
                    } else {
                        &self.costs[idx].1
                    }
                }
            }
        }
    }
}

impl ParsimonyScoring for ModelScoring {
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64 {
        self.scoring(OrderedFloat(blen)).c[(self.alphabet.index(i), self.alphabet.index(j))]
    }

    fn min_match(&self, blen: f64, i: &ParsimonySet, j: &ParsimonySet) -> f64 {
        i.iter()
            .zip(j.iter())
            .map(|(a, b)| self.r#match(blen, a, b))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
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
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use std::fmt::Debug;

    use approx::assert_relative_eq;

    use crate::substitution_models::*;

    use super::*;
    use super::{DiagonalZeros as Z, ModelScoringBuilder as MCB, Rounding as R};

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

    #[cfg(test)]
    fn builder_template<P: ParsimonyModel + Clone + PartialEq + Debug>(
        model: P,
        diagonal: Z,
        rounding: R,
    ) {
        let gap = GapCost::new(6.5, 3.0);
        let builder = ModelScoringBuilder::new(model.clone())
            .gap_cost(gap)
            .diagonal(diagonal)
            .rounding(rounding)
            .times(vec![0.1, 0.5, 10.0]);

        assert_eq!(builder.model, model);
        assert_eq!(builder.gap, gap);
        assert_eq!(builder.diagonal, diagonal);
        assert_eq!(builder.rounding, rounding);
    }

    #[test]
    fn model_builder_setters() {
        for d in [Z::zero(), Z::non_zero()] {
            for r in [R::zero(), R::four(), R::none()] {
                builder_template(SubstModel::<HIVB>::new(&[], &[]), d, r);
                builder_template(SubstModel::<WAG>::new(&[], &[]), d, r);
                builder_template(SubstModel::<BLOSUM>::new(&[], &[]), d, r);

                builder_template(SubstModel::<JC69>::new(&[], &[]), d, r);
                builder_template(SubstModel::<K80>::new(&[], &[]), d, r);
                builder_template(
                    SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]),
                    d,
                    r,
                );
                builder_template(
                    SubstModel::<TN93>::new(
                        &[0.22, 0.26, 0.33, 0.19],
                        &[0.5970915, 0.2940435, 0.00135],
                    ),
                    d,
                    r,
                );
                builder_template(
                    SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]),
                    d,
                    r,
                );
            }
        }
    }

    #[cfg(test)]
    fn builder_default_template<P: ParsimonyModel + Clone + PartialEq + Debug>(model: P) {
        use crate::parsimony::scoring::GapCost;

        let builder = ModelScoringBuilder::new(model.clone());

        assert_eq!(builder.model, model);
        assert_eq!(builder.gap, GapCost::new(2.5, 1.0));
        assert_eq!(builder.diagonal, Z::non_zero());
        assert_eq!(builder.rounding, R::none());
        assert_eq!(builder.times, Vec::<f64>::new());
    }

    #[test]
    fn model_builder_defaults() {
        builder_default_template(SubstModel::<HIVB>::new(&[], &[]));
        builder_default_template(SubstModel::<WAG>::new(&[], &[]));
        builder_default_template(SubstModel::<BLOSUM>::new(&[], &[]));

        builder_default_template(SubstModel::<JC69>::new(&[], &[]));
        builder_default_template(SubstModel::<K80>::new(&[], &[]));
        builder_default_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]));
        builder_default_template(SubstModel::<TN93>::new(
            &[0.22, 0.26, 0.33, 0.19],
            &[0.5970915, 0.2940435, 0.00135],
        ));
        builder_default_template(SubstModel::<GTR>::new(
            &[0.1, 0.3, 0.4, 0.2],
            &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        ));
    }

    #[test]
    fn protein_scorings() {
        let s = ModelScoringBuilder::new(SubstModel::<WAG>::new(&[], &[]))
            .gap_cost(GapCost::new(2.5, 1.0))
            .diagonal(Z::non_zero())
            .rounding(R::zero())
            .times(vec![0.1, 0.3, 0.5, 0.7])
            .build()
            .unwrap();

        let mat_01 = s.scoring(OrderedFloat(0.1));
        let true_matrix_01 = CostMatrix::from_row_slice(20, 20, &TRUE_COST_MATRIX);
        assert_relative_eq!(mat_01.c, true_matrix_01);
        assert_relative_eq!(mat_01.avg, 5.7675);
        assert_relative_eq!(s.avg(0.3), 4.7475);
        assert_relative_eq!(s.avg(0.5), 4.2825);
        assert_relative_eq!(s.avg(0.7), 4.0075);
    }

    #[cfg(test)]
    fn rounding_template<P: ParsimonyModel + Clone>(model: P) {
        let gap = GapCost::new(2.5, 1.0);
        let s_rounded = MCB::new(model.clone())
            .gap_cost(gap)
            .diagonal(Z::zero())
            .rounding(R::zero())
            .times(vec![0.1])
            .build()
            .unwrap();
        let s = MCB::new(model)
            .gap_cost(gap)
            .diagonal(Z::zero())
            .times(vec![0.1])
            .build()
            .unwrap();

        assert_ne!(s_rounded.avg(0.1), s.avg(0.1));
        let rounded_costs = s_rounded.scoring(OrderedFloat(0.1));
        let costs = s.scoring(OrderedFloat(0.1));

        assert_ne!(rounded_costs.c, costs.c);
        for (&e1, &e2) in rounded_costs.c.iter().zip(costs.c.iter()) {
            assert_relative_eq!(e1, e1.round());
            assert_relative_eq!(e1, e2.round());
        }
    }

    #[test]
    fn matrix_entry_rounding() {
        rounding_template(SubstModel::<HIVB>::new(&[], &[]));
        rounding_template(SubstModel::<WAG>::new(&[], &[]));
        rounding_template(SubstModel::<BLOSUM>::new(&[], &[]));

        rounding_template(SubstModel::<JC69>::new(&[], &[]));
        rounding_template(SubstModel::<K80>::new(&[], &[]));
        rounding_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]));
        rounding_template(SubstModel::<TN93>::new(
            &[0.22, 0.26, 0.33, 0.19],
            &[0.5970915, 0.2940435, 0.00135],
        ));
        rounding_template(SubstModel::<GTR>::new(
            &[0.1, 0.3, 0.4, 0.2],
            &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        ));
    }

    #[cfg(test)]
    fn matrix_zero_diagonals_template<P: ParsimonyModel + Clone>(model: P) {
        let gap = GapCost::new(2.5, 1.0);
        let s_zero_diags = MCB::new(model.clone())
            .gap_cost(gap)
            .diagonal(Z::zero())
            .times(vec![0.1])
            .build()
            .unwrap();

        let s = MCB::new(model)
            .gap_cost(gap)
            .times(vec![0.1])
            .build()
            .unwrap();

        assert_ne!(s_zero_diags.avg(0.1), s.avg(0.1));
        assert!(s_zero_diags.avg(0.1) < s.avg(0.1));

        let zero_diag_costs = s_zero_diags.scoring(OrderedFloat(0.1));
        let costs = s.scoring(OrderedFloat(0.1));
        assert_ne!(zero_diag_costs, costs);

        for (&e1, &e2) in zero_diag_costs
            .c
            .diagonal()
            .iter()
            .zip(costs.c.diagonal().iter())
        {
            assert_eq!(e1, 0.0);
            assert_ne!(e1, e2);
            assert_ne!(e2, 0.0);
        }
    }

    #[test]
    fn matrix_zero_diagonals() {
        matrix_zero_diagonals_template(SubstModel::<HIVB>::new(&[], &[]));
        matrix_zero_diagonals_template(SubstModel::<WAG>::new(&[], &[]));
        matrix_zero_diagonals_template(SubstModel::<BLOSUM>::new(&[], &[]));

        matrix_zero_diagonals_template(SubstModel::<JC69>::new(&[], &[]));
        matrix_zero_diagonals_template(SubstModel::<K80>::new(&[], &[]));
        matrix_zero_diagonals_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]));
        matrix_zero_diagonals_template(SubstModel::<TN93>::new(
            &[0.22, 0.26, 0.33, 0.19],
            &[0.5970915, 0.2940435, 0.00135],
        ));
        matrix_zero_diagonals_template(SubstModel::<GTR>::new(
            &[0.1, 0.3, 0.4, 0.2],
            &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        ));
    }
}
