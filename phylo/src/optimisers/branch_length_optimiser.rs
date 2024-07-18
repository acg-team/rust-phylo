use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info, warn};

use crate::substitution_models::{SubstitutionLikelihoodCost, SubstitutionModel};
use crate::tree::NodeIdx;
use crate::Result;

pub(crate) struct SingleBranchOptimiser<'a> {
    pub(crate) likelihood_cost: &'a SubstitutionLikelihoodCost<'a, 4>,
    pub(crate) branch_length: &'a NodeIdx,
    pub(crate) model: &'a SubstitutionModel<4>,
}

impl CostFunction for SingleBranchOptimiser<'_> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut upd_tree = self.likelihood_cost.info.tree.clone();
        upd_tree.set_branch_length(*self.branch_length, *value);
        // self.likelihood_cost.info.tree = upd_tree;
        Ok(-self.likelihood_cost.compute_log_likelihood(self.model).0)
    }

    fn parallelize(&self) -> bool {
        true
    }
}
