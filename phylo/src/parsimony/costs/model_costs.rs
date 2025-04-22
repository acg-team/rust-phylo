use log::{debug, info};
use ordered_float::OrderedFloat;

use crate::alphabets::Alphabet;
use crate::parsimony::{
    CostMatrix, DiagonalZeros, GapCost, ParsimonyCosts, ParsimonyModel, Rounding,
};
use crate::Result;

pub struct ModelCostBuilder {
    model: Box<dyn ParsimonyModel>,
    gap: GapCost,
    diagonal: DiagonalZeros,
    rounding: Rounding,
    times: Vec<f64>,
}

impl ModelCostBuilder {
    pub fn new(model: Box<dyn ParsimonyModel>) -> Self {
        ModelCostBuilder {
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

    pub fn diagonal_zeros(mut self, diagonal: DiagonalZeros) -> Self {
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

    pub fn build(self) -> Result<ModelCosts> {
        ModelCosts::new(
            &*self.model,
            self.gap,
            self.diagonal,
            self.rounding,
            &self.times,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelCosts {
    alphabet: Alphabet,
    costs: Vec<(OrderedFloat<f64>, TimeCosts)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimeCosts {
    avg: f64,
    gap: GapCost,
    c: CostMatrix,
}

impl ModelCosts {
    pub(crate) fn new(
        model: &dyn ParsimonyModel,
        gap: GapCost,
        diagonal: DiagonalZeros,
        rounding: Rounding,
        times: &[f64],
    ) -> Result<Self> {
        info!("Setting up the parsimony scoring from the {} model.", model);

        let mut times = times
            .iter()
            .cloned()
            .map(OrderedFloat::<f64>::from)
            .collect::<Vec<_>>();
        times.sort();

        let costs: Vec<(OrderedFloat<f64>, TimeCosts)> = times
            .iter()
            .map(|time| {
                let cost_matrix = model.scoring(f64::from(*time), &diagonal, &rounding);
                let avg = cost_matrix.mean();
                (
                    *time,
                    TimeCosts {
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
            alphabet: *model.alphabet(),
            costs,
        })
    }

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

    use crate::substitution_models::*;

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
    fn protein_scorings() {
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

    #[cfg(test)]
    fn rounding_template<P: ParsimonyModel>(model: P) {
        let g = GapCost::new(2.5, 1.0);
        let t = OrderedFloat(0.1);
        let s_rounded =
            ModelCosts::new(&model, g.clone(), Z::zero(), R::zero(), &[0.1, 0.4, 0.2]).unwrap();
        let s = ModelCosts::new(&model, g.clone(), Z::zero(), R::none(), &[0.3, 0.1, 0.6]).unwrap();
        assert_ne!(s_rounded.avg(0.1), s.avg(0.1));
        assert_ne!(s_rounded.scoring(t), s.scoring(t));
        for (&e1, &e2) in s_rounded.scoring(t).c.iter().zip(s.scoring(t).c.iter()) {
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
    fn matrix_zero_diagonals_template<P: ParsimonyModel>(model: P) {
        let g = GapCost::new(2.5, 1.0);
        let t = OrderedFloat(0.1);
        let s_zero_diags =
            ModelCosts::new(&model, g.clone(), Z::zero(), R::none(), &[0.1, 0.4, 0.2]).unwrap();
        let s = ModelCosts::new(&model, g.clone(), Z::non_zero(), R::none(), &[0.3, 0.1]).unwrap();
        assert_ne!(s_zero_diags.avg(0.1), s.avg(0.1));
        assert!(s_zero_diags.avg(0.1) < s.avg(0.1));
        assert_ne!(
            s_zero_diags.scoring(OrderedFloat(0.1)),
            s.scoring(OrderedFloat(0.1))
        );
        for (&e1, &e2) in s_zero_diags
            .scoring(t)
            .c
            .diagonal()
            .iter()
            .zip(s.scoring(t).c.diagonal().iter())
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
