use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::PhyloOptimisationResult;
use crate::tree::NodeIdx;
use crate::Result;

pub struct BranchOptimiser<C: TreeSearchCost + Display + Clone> {
    pub(crate) epsilon: f64,
    pub(crate) c: RefCell<C>,
}

impl<C: TreeSearchCost + Clone + Display> BranchOptimiser<C> {
    pub fn new(cost: C) -> Self {
        Self {
            epsilon: 1e-3,
            c: RefCell::new(cost),
        }
    }

    pub fn run(mut self) -> Result<PhyloOptimisationResult<C>> {
        info!("Optimising branch lengths.");
        let mut tree = self.c.borrow().tree().clone();
        let initial_logl = self.c.borrow().cost();

        info!("Initial logl: {}.", initial_logl);
        let mut curr_cost = initial_logl;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        let nodes: Vec<NodeIdx> = tree.iter().map(|node| node.idx).collect();
        while (curr_cost - prev_cost) > self.epsilon {
            iterations += 1;
            info!("Iteration: {}, current logl: {}.", iterations, curr_cost);
            prev_cost = curr_cost;
            for branch in &nodes {
                if tree.root == *branch {
                    continue;
                }
                debug!("Node {:?}: optimising", branch);
                let (logl, length) = self.optimise_branch(branch)?;
                if logl > curr_cost {
                    curr_cost = logl;
                    tree.set_blen(branch, length);
                    debug!("    Optimised to {:.5} with logl {:.5}", length, curr_cost);
                }
                // The branch length may have changed during the optimisation attempt, so the tree
                // should be reset even if the optimisation was unsuccessful.
                self.c.borrow_mut().update_tree(tree.clone(), &[*branch]);
            }
        }
        info!("Done optimising branch lengths.");
        info!(
            "Final logl: {}, achieved in {} iteration(s).",
            curr_cost, iterations
        );

        Ok(PhyloOptimisationResult {
            initial_logl,
            final_logl: curr_cost,
            iterations,
            cost: self.c.into_inner(),
        })
    }
}

impl<C: TreeSearchCost + Clone + Display> BranchOptimiser<C> {
    pub(crate) fn optimise_branch(&mut self, branch: &NodeIdx) -> Result<(f64, f64)> {
        let start_blen = self.c.borrow().tree().node(branch).blen;
        let (start, end) = if start_blen == 0.0 {
            (0.0, 1.0)
        } else {
            (start_blen * 0.1, start_blen * 10.0)
        };
        let optimiser = SingleBranchOptimiser {
            cost: &mut self.c,
            branch: *branch,
        };
        let gss = BrentOpt::new(start, end);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_blen))
            .run()?;
        let state = res.state();
        Ok((-state.best_cost, state.best_param.unwrap()))
    }
}

pub(crate) struct SingleBranchOptimiser<'a, C: TreeSearchCost> {
    pub(crate) cost: &'a RefCell<C>,
    pub(crate) branch: NodeIdx,
}

impl<C: TreeSearchCost> CostFunction for SingleBranchOptimiser<'_, C> {
    type Param = f64;
    type Output = f64;

    fn cost(&self, value: &f64) -> Result<f64> {
        let mut tree = self.cost.borrow().tree().clone();
        tree.set_blen(&self.branch, *value);
        self.cost.borrow_mut().update_tree(tree, &[self.branch]);
        Ok(-self.cost.borrow().cost())
    }

    fn parallelize(&self) -> bool {
        true
    }
}
