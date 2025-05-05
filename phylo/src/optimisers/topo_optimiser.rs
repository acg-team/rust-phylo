use std::fmt::Display;
use std::num::NonZeroUsize;

use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, PhyloOptimisationResult};
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub enum TopologyOptimiserPredicate {
    GtEpsilon(f64),
    FixedIter(NonZeroUsize),
    // NOTE: use of `fn(..) -> ..` disallows closures that capture any
    // surrounding variables, for that we would need to allow Boxed Fn
    // trait objects (or introduce a generic parameter which might get tedious)
    Custom(fn(usize, f64) -> bool),
}

impl TopologyOptimiserPredicate {
    fn test(&self, iteration: usize, delta: f64) -> bool {
        use TopologyOptimiserPredicate::*;
        match *self {
            GtEpsilon(min_delta) => delta > min_delta,
            FixedIter(max) => max.get() > iteration,
            Custom(pred) => pred(iteration, delta),
        }
    }
    pub fn gt_epsilon(epsilon: f64) -> Self {
        Self::GtEpsilon(epsilon)
    }
    pub fn fixed_iter(num: NonZeroUsize) -> Self {
        Self::FixedIter(num)
    }
    pub fn custom(pred: fn(usize, f64) -> bool) -> Self {
        Self::Custom(pred)
    }
}

pub struct TopologyOptimiser<C: TreeSearchCost + Display + Clone> {
    pub(crate) predicate: TopologyOptimiserPredicate,
    pub(crate) c: C,
}

impl<C: TreeSearchCost + Clone + Display> TopologyOptimiser<C> {
    pub fn new(cost: C) -> Self {
        Self {
            predicate: TopologyOptimiserPredicate::GtEpsilon(1e-3),
            c: cost,
        }
    }
    pub fn new_with_pred(cost: C, predicate: TopologyOptimiserPredicate) -> Self {
        Self { predicate, c: cost }
    }

    pub fn run(mut self) -> Result<PhyloOptimisationResult<C>> {
        debug_assert!(self.c.tree().len() > 3);

        info!("Optimising tree topology with SPRs.");
        let init_cost = self.c.cost();
        let init_tree = self.c.tree();

        info!("Initial cost: {}.", init_cost);
        debug!("Initial tree: \n{}", init_tree);
        let mut curr_cost = init_cost;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        let possible_prunes: Vec<_> = init_tree.find_possible_prune_locations().copied().collect();
        let current_prunes: Vec<_> = possible_prunes.iter().collect();
        cfg_if::cfg_if! {
        if #[cfg(not(feature = "deterministic"))] {
            let mut current_prunes = current_prunes;
            // TODO: decide on an explicit and consistent RNG to use throughout the project
            let rng = &mut rand::thread_rng();
        }
        }

        // The best move on this iteration might still be worse than the current tree, in which case
        // the search stops.
        // This means that curr_cost is always hugher than or equel to prev_cost.
        while self.predicate.test(iterations, curr_cost - prev_cost) {
            iterations += 1;
            info!("Iteration: {}, current cost: {}.", iterations, curr_cost);
            prev_cost = curr_cost;

            #[cfg(not(feature = "deterministic"))]
            {
                use rand::seq::SliceRandom;
                current_prunes.shuffle(rng);
            }

            curr_cost = spr::fold_improving_moves(&mut self.c, curr_cost, &current_prunes)?;

            // Optimise branch lengths on current tree to match PhyML
            let o = BranchOptimiser::new(self.c.clone()).run()?;
            if o.final_cost > curr_cost {
                curr_cost = o.final_cost;
                self.c.update_tree(o.cost.tree().clone(), &[]);
            }
            debug!("Tree after iteration {}: \n{}", iterations, self.c.tree());
        }

        debug_assert_eq!(curr_cost, self.c.cost());
        info!("Done optimising tree topology.");
        info!(
            "Final cost: {}, achieved in {} iteration(s).",
            curr_cost, iterations
        );
        Ok(PhyloOptimisationResult {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            cost: self.c,
        })
    }
}

pub mod spr {
    use std::fmt::Display;

    use itertools::Itertools;
    use log::info;

    use crate::{likelihood::TreeSearchCost, optimisers::RegraftOptimiser, tree::NodeIdx, Result};

    /// Iterates over `prune_locations` in order and applies the best (improving)
    /// SPR move for each pruneing location in place
    /// # Returns:
    /// - the new cost (or `base_cost` if no improvement was found)
    pub fn fold_improving_moves<C: TreeSearchCost + Display + Clone>(
        cost_fn: &mut C,
        base_cost: f64,
        prune_locations: &[&NodeIdx],
    ) -> Result<f64> {
        debug_assert!(
            {
                let correct_prune_locations =
                    cost_fn.tree().find_possible_prune_locations().collect_vec();
                prune_locations
                    .iter()
                    .all(|prune_location| correct_prune_locations.contains(prune_location))
            },
            "all prune locations must be contained in the tree and valid"
        );

        prune_locations
            .iter()
            .copied()
            .try_fold(base_cost, |base_cost, prune| -> Result<_> {
                let regraft_optimiser = RegraftOptimiser::new(cost_fn, prune);
                let Some(best_regraft_info) =
                    regraft_optimiser.find_max_cost_regraft_for_prune(base_cost)?
                else {
                    return Ok(base_cost);
                };

                let (best_cost, best_regraft, best_tree) = (
                    best_regraft_info.cost(),
                    best_regraft_info.regraft(),
                    best_regraft_info.into_tree(),
                );

                if best_cost > base_cost {
                    cost_fn.update_tree(best_tree, &[*prune, best_regraft]);
                    info!(
                        "    Regrafted to {:?}, new cost {}.",
                        best_regraft, best_cost
                    );
                    Ok(best_cost)
                } else {
                    info!("    No improvement, best cost {}.", best_cost);
                    Ok(base_cost)
                }
            })
    }
}
