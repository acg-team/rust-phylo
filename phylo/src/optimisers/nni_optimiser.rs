use crate::likelihood::TreeSearchCost;
use anyhow::bail;

use crate::optimisers::{MoveCostInfo, TreeMover};
use crate::tree::{
    NodeIdx::{self, Leaf},
    Tree,
};
use crate::Result;

#[derive(Clone)]
pub struct NniOptimiser {}

impl TreeMover for NniOptimiser {
    fn tree_move_at_location<C>(
        &self,
        base_cost: f64,
        cost: &C,
        node: &crate::tree::NodeIdx,
    ) -> crate::Result<Option<MoveCostInfo>>
    where
        C: TreeSearchCost<Self> + std::fmt::Display + Send + Clone + std::fmt::Display,
    {
        // TODO: here is must call the re-estimation of MASA internal nodes
        // also the branch len opti
        // is the cost bound to MSA or MASA?
        Ok(None)
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
