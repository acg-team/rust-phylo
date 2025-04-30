use std::fmt::Display;

use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, PhyloOptimisationResult};
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub struct TopologyOptimiser<C: TreeSearchCost + Display + Clone> {
    pub(crate) epsilon: f64,
    pub(crate) c: C,
}

pub struct RegraftCostInfo {
    pub regraft: NodeIdx,
    pub cost: f64,
    pub tree: Tree,
}

impl<C: TreeSearchCost + Clone + Display> TopologyOptimiser<C> {
    // TODO: make tree search work under parsimony
    pub fn new(cost: C) -> Self {
        Self {
            epsilon: 1e-3,
            c: cost,
        }
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
        while (curr_cost - prev_cost) > self.epsilon {
            iterations += 1;
            info!("Iteration: {}, current cost: {}.", iterations, curr_cost);
            prev_cost = curr_cost;

            #[cfg(not(feature = "deterministic"))]
            {
                use rand::seq::SliceRandom;
                current_prunes.shuffle(rng);
            }

            curr_cost = Self::fold_improving_spr_moves(&mut self.c, curr_cost, &current_prunes)?;

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

    fn find_regraft_options(prune_branch: &NodeIdx, tree: &Tree) -> Vec<NodeIdx> {
        let all_locations = tree.preorder();
        let prune_subtrees = tree.preorder_subroot(prune_branch);
        let sibling = &tree.sibling(prune_branch).unwrap();
        let parent = &tree.node(prune_branch).parent.unwrap();
        all_locations
            .iter()
            .filter(|&n| {
                n != sibling && n != parent && n != &tree.root && !prune_subtrees.contains(n)
            })
            .cloned()
            .collect()
    }

    /// # Returns:
    /// - the new cost (or `curr_cost` if no improvement was found)
    pub fn fold_improving_spr_moves(
        cost_fn: &mut C,
        curr_cost: f64,
        prune_locations: &[&NodeIdx],
    ) -> Result<f64> {
        prune_locations
            .iter()
            .copied()
            .try_fold(curr_cost, |base_cost, prune| -> Result<_> {
                let Some(best_regraft_info) =
                    Self::find_max_cost_regraft_for_prune(prune, base_cost, cost_fn)?
                else {
                    return Ok(base_cost);
                };

                if best_regraft_info.cost > base_cost {
                    cost_fn
                        .update_tree(best_regraft_info.tree, &[*prune, best_regraft_info.regraft]);
                    info!(
                        "    Regrafted to {:?}, new cost {}.",
                        best_regraft_info.regraft, best_regraft_info.cost
                    );
                    Ok(best_regraft_info.cost)
                } else {
                    info!("    No improvement, best cost {}.", best_regraft_info.cost);
                    Ok(base_cost)
                }
            })
    }

    pub fn find_max_cost_regraft_for_prune(
        prune_location: &NodeIdx,
        base_cost: f64,
        cost: &C,
    ) -> Result<Option<RegraftCostInfo>> {
        let tree = cost.tree();
        if tree.children(&tree.root).contains(prune_location) {
            // due to topology change the current node may have become the direct child of root
            return Ok(None);
        }

        let regraft_locations = Self::find_regraft_options(prune_location, tree);

        info!("Node {:?}: trying to regraft", prune_location);
        // NOTE: collecting the moves into a Vec before finding the max instead of iterating
        // over lazy iterators proved to be minutely faster (than both rayon::try_reduce and for
        // loop over std Iterator)
        let moves = Self::regraft_costs(base_cost, *prune_location, regraft_locations, cost)?;

        let best_regraft = moves
            .into_iter()
            .max_by(|left, right| {
                left.cost
                    .partial_cmp(&right.cost)
                    .expect("tree cost should be a number")
            })
            .unwrap();
        Ok(Some(best_regraft))
    }

    fn calc_spr_cost(
        prune_location: NodeIdx,
        regraft: NodeIdx,
        base_cost: f64,
        mut cost_func: C,
    ) -> Result<RegraftCostInfo> {
        let mut new_tree = cost_func.tree().rooted_spr(&prune_location, &regraft)?;

        cost_func.update_tree(new_tree.clone(), &[prune_location, regraft]);

        let mut move_cost = cost_func.cost();
        if move_cost <= base_cost {
            // reoptimise branch length at the regraft location
            let mut o = BranchOptimiser::new(cost_func);
            let blen_opt = o.optimise_branch(&regraft)?;
            if blen_opt.final_cost > move_cost {
                move_cost = blen_opt.final_cost;
                new_tree.set_blen(&regraft, blen_opt.value);
            }
        }
        debug!("    Regraft to {:?} w best cost {}.", regraft, move_cost);
        Ok::<_, anyhow::Error>(RegraftCostInfo {
            cost: move_cost,
            regraft,
            tree: new_tree,
        })
    }

    pub fn regraft_costs(
        base_cost: f64,
        prune_location: NodeIdx,
        regraft_locations: Vec<NodeIdx>,
        cost: &C,
    ) -> Result<Vec<RegraftCostInfo>> {
        regraft_locations
            .into_iter()
            .map(move |regraft| {
                Self::calc_spr_cost(prune_location, regraft, base_cost, cost.clone())
            })
            .collect()
    }
}
