use std::cell::RefCell;
use std::fmt::Display;

use argmin::core::{CostFunction, Executor, IterState, State};
use argmin::solver::brent::BrentOpt;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{PhyloOptimisationResult, SingleValOptResult};
use crate::tree::NodeIdx;
use crate::{Result, MAX_BLEN};

pub struct BranchOptimiser<C: TreeSearchCost + Display + Clone> {
    pub(crate) epsilon: f64,
    // TODO: RefCell probably not needed here
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
        info!("Optimising branch lengths");
        let init_cost = self.c.borrow().cost();
        let mut tree = self.c.borrow().tree().clone();

        info!("Initial cost: {init_cost}");
        let mut curr_cost = init_cost;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        let nodes: Vec<NodeIdx> = tree.iter().map(|node| node.idx).collect();
        while (curr_cost - prev_cost) > self.epsilon {
            iterations += 1;
            info!("Iteration: {iterations}, current cost: {curr_cost}");
            prev_cost = curr_cost;

            for branch in &nodes {
                if tree.root == *branch {
                    continue;
                }
                debug!("Node {branch:?}: optimising branch length");
                let blen_opt = self.optimise_branch(branch)?;
                if blen_opt.final_cost > curr_cost {
                    curr_cost = blen_opt.final_cost;
                    tree.set_blen(branch, blen_opt.value);
                    debug!(
                        "    Optimised to {:.5} with cost {curr_cost:.5}",
                        blen_opt.value
                    );
                }
                // The branch length may have changed during the optimisation attempt, so the tree
                // should be reset even if the optimisation was unsuccessful.
                self.c.borrow_mut().update_tree(tree.clone(), &[*branch]);
            }
        }

        debug_assert_eq!(curr_cost, self.c.borrow().cost());
        info!("Done optimising branch lengths.");
        info!("Final cost: {curr_cost}, achieved in {iterations} iteration(s)");
        Ok(PhyloOptimisationResult {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            cost: self.c.into_inner(),
        })
    }
}

impl<C: TreeSearchCost + Clone + Display> BranchOptimiser<C> {
    pub(crate) fn optimise_branch(&mut self, branch: &NodeIdx) -> Result<SingleValOptResult> {
        let start_blen = self.c.borrow().tree().node(branch).blen;
        let (min, max) = if start_blen == 0.0 {
            (0.0, 1.0)
        } else {
            (start_blen * 0.1, MAX_BLEN.min(start_blen * 10.0))
        };
        let optimiser = SingleBranchOptimiser {
            cost: &mut self.c,
            branch: *branch,
        };
        let gss = BrentOpt::new(min, max);
        let res = Executor::new(optimiser, gss)
            .configure(|_| IterState::new().param(start_blen))
            .run()?;
        let state = res.state();
        Ok(SingleValOptResult {
            final_cost: -state.best_cost,
            value: state.best_param.unwrap(),
        })
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
