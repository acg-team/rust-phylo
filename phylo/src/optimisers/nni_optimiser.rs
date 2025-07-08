use std::f64;
use std::fmt::Display;

use crate::likelihood::TreeSearchCost;
use crate::Result;
use anyhow::bail;

use crate::optimisers::{MoveCostInfo, TreeMover};
use crate::tree::{
    NodeIdx::{self, Leaf},
    Tree,
};

use super::BranchOptimiser;

#[derive(Clone)]
pub struct NniOptimiser {}

impl TreeMover for NniOptimiser {
    fn tree_move_at_location<C>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &crate::tree::NodeIdx,
    ) -> Result<Option<MoveCostInfo>>
    where
        C: TreeSearchCost<Self> + std::fmt::Display + Send + Clone + std::fmt::Display,
    {
        let mut max_cost_info = None;
        let mut max_cost = f64::MIN;
        println!("running nnis at {}, base_cost = {}", node_idx, base_cost);
        for child_idx in &cost.tree().node(node_idx).children {
            // TODO: is this cost.clone() really the way to go?
            let move_cost_info =
                calc_nni_cost_with_blen_opt(node_idx, child_idx, base_cost, cost.clone())?;
            if move_cost_info.cost() > max_cost {
                println!(
                    "   moving child {} increased cost to {}",
                    child_idx,
                    move_cost_info.cost()
                );
                max_cost = move_cost_info.cost();
                max_cost_info = Some(move_cost_info);
            }
        }
        Ok(max_cost_info)
    }

    fn move_locations<'a>(
        &self,
        tree: &'a crate::tree::Tree,
    ) -> impl Iterator<Item = &'a crate::tree::NodeIdx> {
        tree.preorder()
            .iter()
            .filter(|&n| *n != tree.root && !matches!(n, Leaf(_)))
    }
}

fn calc_nni_cost_with_blen_opt<C: TreeSearchCost<TM> + Clone + Display, TM: TreeMover>(
    node_idx: &NodeIdx,
    child_idx: &NodeIdx,
    base_cost: f64,
    mut cost_fn: C,
) -> Result<MoveCostInfo> {
    let mut new_tree = rooted_nni(cost_fn.tree(), node_idx, child_idx)?;
    cost_fn.update_tree(new_tree.clone(), &[*node_idx]);
    let mut move_cost = cost_fn.cost();
    // TODO: do we really want to do run is only when its worse?
    // if that is not the case. do we run this somewhere is (if the move is good in itself (even
    // without blen opti))
    if cost_fn.blen_optimisation() && move_cost <= base_cost {
        println!(
            "      found a nni move, doing blen opti now, cost {}",
            move_cost
        );
        let mut o = BranchOptimiser::new(cost_fn);
        let blen_opt = o.optimise_branch(node_idx)?;
        if blen_opt.final_cost > move_cost {
            move_cost = blen_opt.final_cost;
            new_tree.set_blen(node_idx, blen_opt.value);
        }
    }
    Ok(MoveCostInfo::new(move_cost, new_tree, vec![*node_idx]))
}

pub(crate) fn rooted_nni(tree: &Tree, node_idx: &NodeIdx, child_idx: &NodeIdx) -> Result<Tree> {
    if node_idx == &tree.root {
        bail!("For the rooted NNI the node mustn't be the root of the tree.");
    }
    if matches!(node_idx, Leaf(_)) {
        bail!("For the rooted NNI the node mustn't be a leaf");
    }
    if tree.node(child_idx).parent.is_none() || tree.node(child_idx).parent.unwrap() != *node_idx {
        bail!("The provided child is not the child");
    }

    Ok(rooted_nni_unchecked(tree, node_idx, child_idx))
}

/// .           |
/// .      -- parent --
/// .      |          |
/// .  --node--      sibling
/// .  |      |
/// .  .    child
///     
/// Swapping child with sibling.
pub(crate) fn rooted_nni_unchecked(tree: &Tree, node_idx: &NodeIdx, child_idx: &NodeIdx) -> Tree {
    let mut new_tree = tree.clone();
    let sibling = tree.node(&tree.sibling(node_idx).unwrap());
    let parent = tree.node(&tree.node(node_idx).parent.unwrap());
    let node = tree.node(node_idx);

    new_tree.dirty[usize::from(node_idx)] = true;

    {
        let parent = new_tree.node_mut(&tree.node(node_idx).parent.unwrap());
        parent.children.retain(|c| c == node_idx);
        parent.children.push(*child_idx);
    };

    {
        let child = new_tree.node_mut(child_idx);
        child.parent = Some(parent.idx);
    };

    {
        let node = new_tree.node_mut(node_idx);
        node.children.retain(|c| c != child_idx);
        node.children.push(sibling.idx);
    };

    {
        let sibling = new_tree.node_mut(&sibling.idx);
        sibling.parent = Some(node.idx);
    };

    new_tree.compute_preorder();
    new_tree.compute_postorder();
    debug_assert_eq!(new_tree.postorder().len(), new_tree.preorder().len());
    debug_assert_eq!(new_tree.postorder().len(), tree.postorder().len());
    new_tree
}
