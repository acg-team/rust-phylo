use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{PhyloOptimisationResult, PhyloOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::tree::NodeIdx;
use crate::Result;

pub(crate) struct SingleBranchOptimiser<'a> {
    pub(crate) c: &'a dyn PhyloCostFunction,
    pub(crate) info: &'a PhyloInfo,
    pub(crate) branch: &'a NodeIdx,
}

impl<'a> CostFunction for SingleBranchOptimiser<'a> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &Self::Param) -> Result<Self::Output> {
        let mut info = self.info.clone();
        info.tree.set_branch_length(self.branch, *value);
        Ok(-self.c.cost(&info))
    }

    fn parallelize(&self) -> bool {
        true
    }
}

pub struct BranchOptimiser<'a> {
    pub(crate) epsilon: f64,
    pub(crate) c: &'a dyn PhyloCostFunction,
    pub(crate) info: PhyloInfo,
}

impl<'a> PhyloOptimiser<'a> for BranchOptimiser<'a> {
    fn new(cost: &'a dyn PhyloCostFunction, phylo_info: &PhyloInfo) -> Self {
        BranchOptimiser {
            epsilon: 1e-3,
            c: cost,
            info: phylo_info.clone(),
        }
    }

    fn run(self) -> Result<PhyloOptimisationResult> {
        info!("Optimising branch lengths.");
        let mut info = self.info.clone();

        let initial_logl = self.c.cost(&info);
        info!("Initial logl: {}.", initial_logl);
        let mut final_logl = initial_logl;
        let mut prev_logl = f64::NEG_INFINITY;
        let mut iterations = 0;

        let nodes: Vec<NodeIdx> = info.tree.iter().map(|node| node.idx).collect();
        while (prev_logl - final_logl).abs() > self.epsilon {
            iterations += 1;
            debug!("Iteration: {}", iterations);
            prev_logl = final_logl;
            for branch in &nodes {
                if info.tree.root == *branch {
                    continue;
                }
                let (logl, length) = self.optimise_branch(branch, &info)?;
                if logl < final_logl {
                    continue;
                }
                final_logl = logl;
                info.tree.set_branch_length(branch, length);
                debug!(
                    "Optimised {} branch length to value {:.5} with logl {:.5}",
                    branch, length, final_logl
                );
            }
        }
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            final_logl, iterations
        );
        Ok(PhyloOptimisationResult {
            initial_logl,
            final_logl,
            iterations,
            tree: info.tree.clone(),
            alignment: info.msa.clone(),
        })
    }
}

impl<'a> BranchOptimiser<'a> {
    fn optimise_branch(&self, branch: &NodeIdx, info: &PhyloInfo) -> Result<(f64, f64)> {
        let start_blen = info.tree.blen(branch);
        let (start, end) = if start_blen == 0.0 {
            (0.0, 1.0)
        } else {
            (start_blen * 0.1, start_blen * 10.0)
        };
        let optimiser = SingleBranchOptimiser {
            c: self.c,
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