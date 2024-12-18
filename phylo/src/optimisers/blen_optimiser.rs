use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{PhyloOptimisationResult, PhyloOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::tree::NodeIdx;
use crate::Result;

pub(crate) struct SingleBranchOptimiser<'a, EM: PhyloCostFunction> {
    pub(crate) model: &'a EM,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) branch: &'a NodeIdx,
}

impl<EM: PhyloCostFunction> CostFunction for SingleBranchOptimiser<'_, EM> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        let mut info = self.info.clone();
        info.tree.set_blen(self.branch, *value);
        Ok(-self.model.cost(&info, false))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct BranchOptimiser<'a, EM: PhyloCostFunction> {
    pub(crate) epsilon: f64,
    pub(crate) model: &'a EM,
    pub(crate) info: PhyloInfo,
}

impl<'a, EM: PhyloCostFunction + Clone> PhyloOptimiser<'a, EM> for BranchOptimiser<'a, EM> {
    fn new(model: &'a EM, info: &PhyloInfo) -> Self {
        BranchOptimiser {
            epsilon: 1e-3,
            model,
            info: info.clone(),
        }
    }

    fn run(self) -> Result<PhyloOptimisationResult> {
        info!("Optimising branch lengths.");
        let mut info = self.info.clone();
        let initial_logl = self.model.cost(&info, true);

        info!("Initial logl: {}.", initial_logl);
        let mut curr_cost = initial_logl;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        let nodes: Vec<NodeIdx> = info.tree.iter().map(|node| node.idx).collect();
        while (curr_cost - prev_cost) > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_cost = curr_cost;
            for branch in &nodes {
                if info.tree.root == *branch {
                    continue;
                }
                let (logl, length) = self.optimise_branch(branch, &info)?;
                if logl > curr_cost {
                    curr_cost = logl;
                    info.tree.set_blen(branch, length);
                    debug!(
                        "Optimised {} branch length to value {:.5} with logl {:.5}",
                        branch, length, curr_cost
                    );
                }
                info.tree.clean(true);
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            curr_cost, iterations
        );
        Ok(PhyloOptimisationResult {
            initial_logl,
            final_logl: curr_cost,
            iterations,
            i: info,
        })
    }
}

impl<EM: PhyloCostFunction> BranchOptimiser<'_, EM> {
    pub(crate) fn optimise_branch(&self, branch: &NodeIdx, info: &PhyloInfo) -> Result<(f64, f64)> {
        let start_blen = info.tree.blen(branch);
        let (start, end) = if start_blen == 0.0 {
            (0.0, 1.0)
        } else {
            (start_blen * 0.1, start_blen * 10.0)
        };
        let optimiser = SingleBranchOptimiser {
            model: self.model,
            info,
            branch,
        };
        let gss = BrentOpt::new(start, end);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_blen))
            .run()?;
        let state = res.state();
        Ok((-state.best_cost, state.best_param.unwrap()))
    }
}
