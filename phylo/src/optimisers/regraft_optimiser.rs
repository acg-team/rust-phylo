use std::fmt::Display;

use itertools::Itertools;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::BranchOptimiser;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub struct RegraftCostInfo {
    regraft: NodeIdx,
    cost: f64,
    tree: Tree,
}

impl RegraftCostInfo {
    pub fn regraft(&self) -> NodeIdx {
        self.regraft
    }
    pub fn cost(&self) -> f64 {
        self.cost
    }
    pub fn tree(&self) -> &Tree {
        &self.tree
    }
    pub fn into_tree(self) -> Tree {
        self.tree
    }
}

pub struct RegraftOptimiser<'a, C: TreeSearchCost + Clone + Display + Send> {
    prune_location: &'a NodeIdx,
    cost_fn: &'a C,
}
impl<'a, C: TreeSearchCost + Clone + Display + Send> RegraftOptimiser<'a, C> {
    pub fn new(cost_fn: &'a C, prune_location: &'a NodeIdx) -> RegraftOptimiser<'a, C> {
        Self {
            prune_location,
            cost_fn,
        }
    }
    pub fn available_regraft_locations(&self) -> impl Iterator<Item = &NodeIdx> + use<'_, C> {
        let tree = self.cost_fn.tree();
        let all_locations = tree.preorder();
        let prune_subtrees = tree.preorder_subroot(self.prune_location);
        let sibling = tree.sibling(self.prune_location).unwrap();
        let parent = tree.node(self.prune_location).parent.unwrap();
        all_locations.iter().filter(move |&node| {
            *node != sibling
                && *node != parent
                && *node != tree.root
                && !prune_subtrees.contains(node)
        })
    }

    pub fn find_max_cost_regraft_for_prune(
        &self,
        base_cost: f64,
    ) -> Result<Option<RegraftCostInfo>> {
        let tree = self.cost_fn.tree();
        if tree.children(&tree.root).contains(self.prune_location) {
            // due to topology change the current node may have become the direct child of root
            return Ok(None);
        }

        let regraft_locations = self.available_regraft_locations().copied().collect_vec();

        info!("Node {:?}: trying to regraft", self.prune_location);
        // NOTE: collecting the moves into a Vec before finding the max instead of iterating
        // over lazy iterators proved to be minutely faster/equal to (both rayon::try_reduce and for
        // loop over std Iterator)
        let moves = calc_regraft_costs(
            base_cost,
            *self.prune_location,
            regraft_locations,
            self.cost_fn,
        )?;

        let best_regraft = moves
            .into_iter()
            .max_by(|left, right| {
                left.cost
                    .partial_cmp(&right.cost)
                    .expect("tree cost should be a number")
            })
            .expect("at least one SPR move has to be evaluated");
        Ok(Some(best_regraft))
    }
}

cfg_if::cfg_if! {
if #[cfg(feature="par-regraft")] {
fn calc_regraft_costs<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<Vec<RegraftCostInfo>> {
    use rayon::prelude::*;
    let cost_funcs = vec![cost.clone(); regraft_locations.len()];
    regraft_locations
        .into_par_iter()
        .zip_eq(cost_funcs)
        .map(move |(regraft, cost_fn)| {
            calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost_fn.clone())
        })
        .collect()
}
} else if #[cfg(feature="par-regraft-chunk")] {
/// NOTE: seems to be faster than full on parallel for few taxa
fn calc_regraft_costs<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<Vec<RegraftCostInfo>> {
    use rayon::prelude::*;
    // TODO: determine better factor (maybe dynamically)
    const CHUNK_SIZE: usize = 2;
    let cost_funcs = vec![cost.clone(); regraft_locations.len().div_ceil(CHUNK_SIZE)];
    regraft_locations
        .par_chunks(CHUNK_SIZE)
        .zip_eq(cost_funcs)
        .map(move |(regrafts, cost_func)| -> Result<_> {
            let mut max: Option<RegraftCostInfo> = None;
            for regraft_result in regrafts.iter().map(move |regraft| {
                calc_spr_cost_with_blen_opt(prune_location, *regraft, base_cost, cost_func.clone())
            }) {
                let regraft = regraft_result?;
                if max.as_ref().is_none_or(|max| max.cost > regraft.cost) {
                    max = Some(regraft);
                }
            }
            Ok(max.expect("at least one regraft location"))
        })
        .collect()
}
} else {
fn calc_regraft_costs<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<Vec<RegraftCostInfo>> {
    regraft_locations.into_iter().map(move |regraft| {
        calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost.clone())
    }).collect()
}
}}

/// if the move doesn't result in improvement over `base_cost`
/// the blen of the regrafted branch is optimised to check if an
/// improvement could still be reached
fn calc_spr_cost_with_blen_opt<C: TreeSearchCost + Clone + Display>(
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
    Ok(RegraftCostInfo {
        cost: move_cost,
        regraft,
        tree: new_tree,
    })
}
