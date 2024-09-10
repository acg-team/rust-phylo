use log::{debug, info};

use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{BranchOptimiser, PhyloOptimisationResult, PhyloOptimiser};
use crate::phylo_info::PhyloInfo;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub struct TopologyOptimiser<'a, EM: PhyloCostFunction> {
    pub(crate) epsilon: f64,
    pub(crate) model: &'a EM,
    pub(crate) info: PhyloInfo,
}

impl<'a, EM: PhyloCostFunction> PhyloOptimiser<'a, EM> for TopologyOptimiser<'a, EM> {
    fn new(model: &'a EM, info: &PhyloInfo) -> Self {
        TopologyOptimiser {
            epsilon: 1e-3,
            model,
            info: info.clone(),
        }
    }

    fn run(self) -> Result<PhyloOptimisationResult> {
        self.model.reset();
        debug_assert!(self.info.tree.len() > 3);
        info!("Optimising tree topology with SPRs.");
        let mut info = self.info.clone();

        let initial_logl = self.model.cost(&info);
        info!("Initial logl: {}.", initial_logl);
        let mut curr_cost = initial_logl;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        // No pruning on the root branch
        let prune_locations: Vec<NodeIdx> = info
            .tree
            .preorder()
            .iter()
            .filter(|&n| n != &info.tree.root)
            .cloned()
            .collect();
        while (curr_cost - prev_cost) > self.epsilon {
            iterations += 1;
            info!("Iteration: {}.", iterations);
            prev_cost = curr_cost;
            info!("Current logl: {}.", curr_cost);
            for prune_branch in &prune_locations {
                if info.tree.children(&info.tree.root).contains(prune_branch) {
                    // due to topology change the current node may have become the direct child of root
                    continue;
                }
                let regraft_locations = Self::find_regraft_options(prune_branch, &info);
                let mut moves: Vec<(f64, Tree)> = Vec::with_capacity(regraft_locations.len());
                for regraft_branch in &regraft_locations {
                    let mut new_info = info.clone();
                    new_info.tree = info.tree.rooted_spr(prune_branch, regraft_branch).unwrap();
                    let mut logl = self.model.cost(&new_info);
                    if logl <= curr_cost {
                        // reoptimise branch lengths at the regraft location
                        let o = BranchOptimiser::new(self.model, &new_info);
                        let (blen_logl, blen) = o.optimise_branch(regraft_branch, &new_info)?;
                        if blen_logl > logl {
                            new_info.tree.set_blen(regraft_branch, blen);
                            logl = blen_logl;
                        }
                    }
                    moves.push((logl, new_info.tree.clone()));
                }
                let (best_logl, best_tree) = moves
                    .into_iter()
                    .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
                    .unwrap();
                if best_logl > curr_cost {
                    curr_cost = best_logl;
                    info.tree = best_tree;
                    debug!("Regrafted {} with new logl {}.", prune_branch, curr_cost);
                } else {
                    debug!(
                        "No improvement regrafting {}, best logl {}.",
                        prune_branch, best_logl
                    );
                }
                // Optimise branch lengths on the final tree
                let o = BranchOptimiser::new(self.model, &info).run()?;
                if o.final_logl > curr_cost {
                    curr_cost = o.final_logl;
                    info = o.i;
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

impl<EM: PhyloCostFunction> TopologyOptimiser<'_, EM> {
    fn find_regraft_options(prune_branch: &NodeIdx, info: &PhyloInfo) -> Vec<NodeIdx> {
        let all_locations = info.tree.preorder();
        let prune_subtrees = info.tree.preorder_subroot(prune_branch);
        let sibling = &info.tree.sibling(prune_branch).unwrap();
        let parent = &info.tree.node(prune_branch).parent.unwrap();
        all_locations
            .iter()
            .filter(|&n| {
                !prune_subtrees.contains(n) && n != sibling && n != parent && n != &info.tree.root
            })
            .cloned()
            .collect()
    }
}
