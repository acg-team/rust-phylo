use std::cell::RefCell;
use std::fmt::Display;

use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, PhyloOptimisationResult};
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub struct TopologyOptimiser<C: TreeSearchCost + Display + Clone> {
    pub(crate) epsilon: f64,
    pub(crate) c: RefCell<C>,
}

impl<C: TreeSearchCost + Clone + Display> TopologyOptimiser<C> {
    // TODO: make tree search work under parsimony
    pub fn new(cost: C) -> Self {
        Self {
            epsilon: 1e-3,
            c: RefCell::new(cost),
        }
    }

    pub fn run(self) -> Result<PhyloOptimisationResult<C>> {
        debug_assert!(self.c.borrow().tree().len() > 3);

        info!("Optimising tree topology with SPRs.");
        let initial_logl = self.c.borrow().cost();
        let mut tree = self.c.borrow().tree().clone();

        info!("Initial logl: {}.", initial_logl);
        debug!("Initial tree: \n{}", tree);
        let mut curr_cost = initial_logl;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        // No pruning on the root branch
        let prune_locations: Vec<NodeIdx> = tree
            .preorder()
            .iter()
            .filter(|&n| n != &tree.root)
            .cloned()
            .collect();

        // The best move on this iteration might still be worse than the current tree, in which case
        // the search stops.
        // This means that curr_cost is always hugher than or equel to prev_cost.
        while (curr_cost - prev_cost) > self.epsilon {
            tree = self.c.borrow().tree().clone();
            iterations += 1;
            info!("Iteration: {}, current logl: {}.", iterations, curr_cost);
            prev_cost = curr_cost;

            for prune in &prune_locations {
                if tree.children(&tree.root).contains(prune) {
                    // due to topology change the current node may have become the direct child of root
                    continue;
                }
                let regraft_locations = Self::find_regraft_options(prune, &tree);
                let mut moves = Vec::<(f64, NodeIdx, Tree)>::with_capacity(regraft_locations.len());

                info!("Node {:?}: trying to regraft", prune);
                for regraft in &regraft_locations {
                    let mut new_tree = tree.rooted_spr(prune, regraft)?;

                    let mut cost = self.c.borrow().clone();
                    cost.update_tree(new_tree.clone(), &[*prune, *regraft]);

                    let mut logl = cost.cost();
                    if logl <= curr_cost {
                        // reoptimise branch length at the regraft location
                        let mut o = BranchOptimiser::new(cost);
                        let (blen_logl, blen) = o.optimise_branch(regraft)?;
                        if blen_logl > logl {
                            new_tree.set_blen(regraft, blen);
                            logl = blen_logl;
                        }
                    }
                    debug!("    Regraft to {:?} w best logl {}.", regraft, logl);
                    moves.push((logl, *regraft, new_tree));
                }
                let (best_logl, regraft, best_tree) = moves
                    .into_iter()
                    .max_by(|(a, _, _), (b, _, _)| a.partial_cmp(b).unwrap())
                    .unwrap();
                if best_logl > curr_cost {
                    curr_cost = best_logl;
                    self.c
                        .borrow_mut()
                        .update_tree(best_tree, &[*prune, regraft]);
                    info!("    Regrafted to {:?}, new logl {}.", regraft, curr_cost);
                } else {
                    info!("    No improvement, best logl {}.", best_logl);
                }
            }

            // Optimise branch lengths on current tree to match PhyML
            let o = BranchOptimiser::new(self.c.borrow().clone()).run()?;
            if o.final_logl > curr_cost {
                curr_cost = o.final_logl;
                self.c.borrow_mut().update_tree(o.cost.tree().clone(), &[]);
            }
            debug!(
                "Tree after iteration {}: \n{}",
                iterations,
                self.c.borrow().tree()
            );
        }

        debug_assert_eq!(curr_cost, self.c.borrow().cost());
        info!("Done optimising tree topology.");
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
}
