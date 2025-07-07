use anyhow::bail;
use approx::relative_eq;
use std::fmt::Display;

use itertools::Itertools;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::tree_mover::{MoveCostInfo, TreeMover};
use crate::optimisers::BranchOptimiser;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

#[derive(Clone)]
pub struct SprOptimiser {}

impl TreeMover for SprOptimiser {
    fn tree_move_at_location<C: TreeSearchCost<Self> + Clone + Display + Send>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &NodeIdx,
    ) -> Result<Option<MoveCostInfo>> {
        self.find_max_cost_regraft_for_prune(base_cost, cost, node_idx)
    }

    fn move_locations<'a>(&self, tree: &'a Tree) -> impl Iterator<Item = &'a NodeIdx> {
        tree.preorder().iter().filter(|&n| *n != tree.root)
    }
}

impl SprOptimiser {
    pub fn available_regraft_locations<'a>(
        &self,
        tree: &'a Tree,
        prune_location: &NodeIdx,
    ) -> impl Iterator<Item = &'a NodeIdx> {
        let all_locations = tree.preorder();
        let prune_subtrees = tree.preorder_subroot(prune_location);
        let sibling = tree.sibling(prune_location).unwrap();
        let parent = tree.node(prune_location).parent.unwrap();
        all_locations.iter().filter(move |&node| {
            *node != sibling
                && *node != parent
                && *node != tree.root
                && !prune_subtrees.contains(node)
        })
    }

    pub fn find_max_cost_regraft_for_prune<
        C: TreeSearchCost<TM> + Clone + Display + Send,
        TM: TreeMover,
    >(
        &self,
        base_cost: f64,
        cost: &C,
        prune_location: &NodeIdx,
    ) -> Result<Option<MoveCostInfo>> {
        let tree = cost.tree();
        if tree.children(&tree.root).contains(prune_location) {
            // due to topology change the current node may have become the direct child of root
            return Ok(None);
        }

        let regraft_locations = self
            .available_regraft_locations(tree, prune_location)
            .copied()
            .collect_vec();

        info!("Node {:?}: trying to regraft", prune_location);
        let best_regraft =
            calc_best_regraft_cost(base_cost, *prune_location, regraft_locations, cost)?;
        Ok(Some(best_regraft))
    }
}

cfg_if::cfg_if! {
if #[cfg(feature="par-regraft")] {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    use rayon::prelude::*;
    let cost_funcs = vec![cost.clone(); regraft_locations.len()];
    regraft_locations
        .into_par_iter()
        .zip_eq(cost_funcs)
        .map(move |(regraft, cost_fn)| {
            calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost_fn.clone())
        })
        .try_reduce_with(|left, right| Ok(if left.cost() > right.cost() {left} else {right})).expect("at least one regraft location")
}
} else if #[cfg(feature="par-regraft-chunk")] {
/// NOTE: seems to be faster than full on parallel for few taxa
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    use rayon::prelude::*;
    // TODO: determine better factor (maybe dynamically)
    const CHUNK_SIZE: usize = 2;
    let cost_funcs = vec![cost.clone(); regraft_locations.len().div_ceil(CHUNK_SIZE)];

    regraft_locations
        .par_chunks(CHUNK_SIZE)
        .zip_eq(cost_funcs)
        .map(move |(regrafts, cost_func)| -> Result<_> {
            let mut max: Option<MoveCostInfo> = None;
            let mut max_cost = f64::MIN;
            for regraft_result in regrafts.iter().map(move |regraft| {
                calc_spr_cost_with_blen_opt(prune_location, *regraft, base_cost, cost_func.clone())
            }) {
                match result {
                    Ok(regraft_info) if regraft_info.cost() > max_cost => {
                        max_cost = regraft_info.cost();
                        max = Some(regraft_info);
                    },
                    Ok(_) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(max.expect("at least one regraft location"))
        })
        .try_reduce_with(|left, right| Ok(if left.cost() > right.cost() {left} else {right})).expect("at least one regraft location")
}
} else if #[cfg(feature="par-regraft-manual")] {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    #[derive(Clone)]
    struct RecursiveForkJoinRegrafter<C: TreeSearchCost + Clone + Display + Send> {
        cost_fn: C,
        prune_location: NodeIdx,
        base_cost: f64,
    }
    /// NOTE: by being recursive these tasks can be stored solely on the stack
    /// using rayon::scope might look simpler but incurs overhead by having to manage
    /// tasks on the heap
    fn regraft_recursive<C: TreeSearchCost + Clone + Display + Send>(state: RecursiveForkJoinRegrafter<C>, regraft_locations: &[NodeIdx]) -> Result<MoveCostInfo> {
        if regraft_locations.len() == 1 {
            return calc_spr_cost_with_blen_opt(state.prune_location, regraft_locations[0], state.base_cost, state.cost_fn);
        }
        let (left_locations, right_locations) = regraft_locations.split_at(regraft_locations.len() / 2);
        let r2 = state.clone();
        match rayon::join(move || regraft_recursive(state, left_locations), move ||regraft_recursive(r2, right_locations)) {
            (Ok(left), Ok(right)) => Ok(if left.cost() > right.cost() {left} else {right}) ,
            (Err(error), _) | (_, Err(error))   => Err(error),
        }
    }
    regraft_recursive(RecursiveForkJoinRegrafter { cost_fn: cost.clone(), prune_location, base_cost }, &regraft_locations)
}
} else {
fn calc_best_regraft_cost<C: TreeSearchCost<TM> + Clone + Display + Send, TM: TreeMover>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost: &C,
) -> Result<MoveCostInfo> {
    let mut max = None;
    let mut max_cost = f64::MIN;
    for regraft in regraft_locations.into_iter().map(move |regraft| {
        calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost.clone())
    }) {
        match regraft {
            Ok(regraft_info) if regraft_info.cost() > max_cost => {
                max_cost = regraft_info.cost();
                max = Some(regraft_info);
            },
            Ok(_) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(max.expect("at least one regraft location"))
}
}}

/// for evo models with branch length optimisation enabled (disabled for parsimony),
/// if the move doesn't result in improvement over `base_cost`
/// the blen of the regrafted branch is optimised to check if an
/// improvement could still be reached
fn calc_spr_cost_with_blen_opt<C: TreeSearchCost<TM> + Clone + Display, TM: TreeMover>(
    prune_location: NodeIdx,
    regraft: NodeIdx,
    base_cost: f64,
    mut cost_fn: C,
) -> Result<MoveCostInfo> {
    let mut new_tree = rooted_spr(cost_fn.tree(), &prune_location, &regraft)?;

    cost_fn.update_tree(new_tree.clone(), &[prune_location, regraft]);

    let mut move_cost = cost_fn.cost();
    if cost_fn.blen_optimisation() && move_cost <= base_cost {
        // reoptimise branch length at the regraft location
        let mut o = BranchOptimiser::new(cost_fn);
        let blen_opt = o.optimise_branch(&regraft)?;
        if blen_opt.final_cost > move_cost {
            move_cost = blen_opt.final_cost;
            new_tree.set_blen(&regraft, blen_opt.value);
        }
    }
    debug!("    Regraft to {:?} w best cost {}.", regraft, move_cost);
    Ok(MoveCostInfo::new(move_cost, new_tree, vec![regraft]))
}

pub fn rooted_spr(tree: &Tree, prune_idx: &NodeIdx, regraft_idx: &NodeIdx) -> Result<Tree> {
    // Prune and regraft nodes must be different
    if prune_idx == regraft_idx {
        bail!("Prune and regraft nodes must be different.");
    }
    if tree.is_subtree(regraft_idx, prune_idx) {
        bail!("Prune node cannot be a subtree of the regraft node.");
    }

    let prune = tree.node(prune_idx);
    // Pruned node must have a parent, it is the one being reattached
    if prune.parent.is_none() {
        bail!("Cannot prune the root node.");
    }
    // Cannot prune direct child of the root node, otherwise branch lengths are undefined
    if tree.node(&prune.parent.unwrap()).parent.is_none() {
        bail!("Cannot prune direct child of the root node.");
    }
    let regraft = tree.node(regraft_idx);
    // Regrafted node must have a parent, the prune parent is attached to that branch
    if regraft.parent.is_none() {
        bail!("Cannot regraft to root node.");
    }
    if regraft.parent == prune.parent {
        bail!("Prune and regraft nodes must have different parents.");
    }

    Ok(rooted_spr_unchecked(tree, prune_idx, regraft_idx))
}

// TODO: alternatively to making this pub(crate) the tests can be moved to the bottom of this
// module
pub(crate) fn rooted_spr_unchecked(
    tree: &Tree,
    prune_idx: &NodeIdx,
    regraft_idx: &NodeIdx,
) -> Tree {
    let prune = tree.node(prune_idx);
    let prune_sib = tree.node(&tree.sibling(&prune.idx).unwrap());
    let prune_par = tree.node(&prune.parent.unwrap());
    let prune_grpar = tree.node(&prune_par.parent.unwrap());
    let regraft = tree.node(regraft_idx);
    let regraft_par = tree.node(&regraft.parent.unwrap());

    let mut new_tree = tree.clone();

    {
        new_tree.dirty[usize::from(prune_sib.idx)] = true;
        new_tree.dirty[usize::from(prune_par.idx)] = true;
    }

    {
        // Sibling of pruned node connects to common parent, branch length is updated
        let prune_sib = new_tree.node_mut(&prune_sib.idx);
        prune_sib.parent = prune_par.parent;
        prune_sib.blen += prune_par.blen;
    };

    {
        // Pruned node's parent is removed from its parent's children, pruned nodes sibling is added
        let prune_grpar = new_tree.node_mut(&prune_grpar.idx);
        prune_grpar.children.retain(|&x| x != prune_par.idx);
        prune_grpar.children.push(prune_sib.idx);
    };

    {
        // Regrafted branch is split in two, parent of regrafted node is now pruned node's parent
        let regraft = new_tree.node_mut(&regraft.idx);
        regraft.parent = Some(prune_par.idx);
        regraft.blen /= 2.0;
    }

    {
        // Regrafted node is removed from its parent's children, pruned node's parent is added
        let prune_par = new_tree.node_mut(&prune_par.idx);
        prune_par.children.retain(|&x| x != prune_sib.idx);
        prune_par.children.push(regraft.idx);
        prune_par.blen = regraft.blen / 2.0;
        prune_par.parent = regraft.parent;
    }

    {
        // Regrafted node's parent's children are updated
        let regraft_par = new_tree.node_mut(&regraft_par.idx);
        regraft_par.children.retain(|&x| x != regraft.idx);
        regraft_par.children.push(prune_par.idx);
    }

    // Tree height should not have changed
    debug_assert!(relative_eq!(
        new_tree.height,
        new_tree.nodes.iter().map(|node| node.blen).sum(),
        epsilon = 1e-10
    ));

    new_tree.compute_postorder();
    new_tree.compute_preorder();
    debug_assert_eq!(new_tree.postorder().len(), tree.postorder().len());
    new_tree
}
