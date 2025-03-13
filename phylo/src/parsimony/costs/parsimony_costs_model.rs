use std::collections::HashMap;

use log::{debug, info};

use crate::alphabets::Alphabet;
use crate::evolutionary_models::EvoModel;
use crate::parsimony::{BranchParsimonyCosts, GapMultipliers, ParsimonyCosts, Rounding};
use crate::substitution_models::ParsimonyModel;
use crate::{Result, cmp_f64, ord_f64};

use super::CostMatrix;

#[derive(Clone, Debug, PartialEq)]
pub struct ParsimonyCostsWModel<M: ParsimonyModel> {
    subst_model: M,
    times: Vec<f64>,
    costs: HashMap<ord_f64, BranchCostsWModel>,
}

impl<M: ParsimonyModel + EvoModel> ParsimonyCostsWModel<M> {
    pub fn new(
        model: M,
        times: &[f64],
        zero_diag: bool,
        gap_mult: &GapMultipliers,
        rounding: &Rounding,
    ) -> Result<Self> {
        info!(
            "Setting up the parsimony scoring from the {} substitution model.",
            model
        );
        let costs = model
            .generate_scorings(times, zero_diag, rounding)
            .into_iter()
            .map(|(key, (branch_costs, avg_cost))| {
                debug!("Average cost for time {} is {}", key, avg_cost);
                debug!(
                    "Gap open cost for time {} is {}",
                    key,
                    gap_mult.open * avg_cost
                );
                debug!(
                    "Gap ext cost for time {} is {}",
                    key,
                    gap_mult.ext * avg_cost
                );
                (
                    key,
                    BranchCostsWModel {
                        avg_cost,
                        gap_open: gap_mult.open * avg_cost,
                        gap_ext: gap_mult.ext * avg_cost,
                        costs: branch_costs,
                        alphabet: model.alphabet().clone(),
                    },
                )
            })
            .collect();

        info!(
            "Created scoring matrices from the {} substitution model for {:?} branch lengths.",
            model, times
        );
        debug!("The scoring matrices are: {:?}", costs);
        Ok(ParsimonyCostsWModel {
            subst_model: model,
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

impl<M: ParsimonyModel> ParsimonyCostsWModel<M> {
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

impl<M: ParsimonyModel + EvoModel> ParsimonyCosts for ParsimonyCostsWModel<M> {
    fn branch_costs(&self, branch_length: f64) -> &dyn BranchParsimonyCosts {
        &self.costs[&ord_f64::from(self.find_closest_branch_length(branch_length))]
    }

    fn alphabet(&self) -> &Alphabet {
        self.subst_model.alphabet()
    }

    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64 {
        self.costs[&ord_f64::from(self.find_closest_branch_length(blen))].r#match(i, j)
    }

    fn gap_open(&self, blen: f64) -> f64 {
        self.costs[&ord_f64::from(self.find_closest_branch_length(blen))].gap_open()
    }

    fn gap_ext(&self, blen: f64) -> f64 {
        self.costs[&ord_f64::from(self.find_closest_branch_length(blen))].gap_ext()
    }

    fn avg(&self, blen: f64) -> f64 {
        self.costs
            .get(&ord_f64::from(self.find_closest_branch_length(blen)))
            .unwrap()
            .avg_cost
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BranchCostsWModel {
    alphabet: Alphabet,
    avg_cost: f64,
    gap_open: f64,
    gap_ext: f64,
    costs: CostMatrix,
}

impl BranchParsimonyCosts for BranchCostsWModel {
    fn r#match(&self, i: &u8, j: &u8) -> f64 {
        self.costs[(self.alphabet.index(i), self.alphabet.index(j))]
    }

    fn gap_ext(&self) -> f64 {
        self.gap_ext
    }

    fn gap_open(&self) -> f64 {
        self.gap_open
    }

    fn avg(&self) -> f64 {
        self.avg_cost
    }

    fn cost_matrix(&self) -> &CostMatrix {
        &self.costs
    }
}
