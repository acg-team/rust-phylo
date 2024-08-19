use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::PhyloInfo;
use crate::tree::NodeIdx;
use crate::Result;

use super::PhyloOptimisationResult;

pub(crate) struct SingleBranchOptimiser<'a> {
    pub(crate) cost: &'a dyn LikelihoodCostFunction,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) branch: &'a NodeIdx,
}

impl<'a> CostFunction for SingleBranchOptimiser<'a> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut info = self.info.clone();
        info.tree.set_branch_length(self.branch, *value);
        Ok(-self.cost.logl(&info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct BranchOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) cost: &'a dyn LikelihoodCostFunction,
    pub(crate) info: &'a PhyloInfo,
}

impl<'a> BranchOptimiser<'a> {
    pub fn new(cost: &'a dyn LikelihoodCostFunction, phylo_info: &'a PhyloInfo) -> Self {
        BranchOptimiser {
            epsilon: 1e-3,
            cost,
            info: phylo_info,
        }
    }

    pub fn run(&self) -> Result<PhyloOptimisationResult> {
        info!("Optimising branch lengths.");
        let mut info = self.info.clone();

        let init_logl = self.cost.logl(&info);
        info!("Initial logl: {}.", init_logl);
        let mut opt_logl = init_logl;
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iters = 0;

        let nodes: Vec<NodeIdx> = info.tree.iter().map(|node| node.idx).collect();
        while (prev_logl - opt_logl).abs() > self.epsilon {
            iters += 1;
            debug!("Iteration: {}", iters);
            prev_logl = opt_logl;
            for branch in &nodes {
                if info.tree.root == *branch {
                    continue;
                }
                let (logl, length) = self.optimise_branch(branch, &info)?;
                if logl < opt_logl {
                    continue;
                }
                opt_logl = logl;
                info.tree.set_branch_length(branch, length);
                debug!(
                    "Optimised {} branch length to value {:.5} with logl {:.5}",
                    branch, length, opt_logl
                );
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            opt_logl, iters
        );
        Ok(PhyloOptimisationResult {
            initial_logl: init_logl,
            final_logl: opt_logl,
            iterations: iters,
            tree: info.tree.clone(),
            alignment: info.msa.clone(),
        })
    }

    fn optimise_branch(&self, branch: &NodeIdx, info: &PhyloInfo) -> Result<(f64, f64)> {
        let start_blen = info.tree.blen(branch);
        let (start, end) = if start_blen == 0.0 {
            (0.0, 1.0)
        } else {
            (start_blen * 0.1, start_blen * 10.0)
        };
        let optimiser = SingleBranchOptimiser {
            cost: self.cost,
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
