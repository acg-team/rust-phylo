use std::f64;
use std::fmt::Display;

use anyhow::bail;

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, MoveCostInfo, MoveOptimiser};
use crate::tree::{
    NodeIdx::{self, Leaf},
    Tree,
};
use crate::Result;

#[derive(Clone)]
pub struct NniOptimiser {}

impl MoveOptimiser for NniOptimiser {
    fn move_locations<'a, C: TreeSearchCost + Display + Send + Clone + Display>(
        &self,
        cost: &'a C,
    ) -> impl Iterator<Item = &'a crate::tree::NodeIdx> {
        cost.tree()
            .preorder()
            .iter()
            .filter(|&n| *n != cost.tree().root && !matches!(n, Leaf(_)))
    }

    fn best_move_at_location<C>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &crate::tree::NodeIdx,
    ) -> Result<Option<MoveCostInfo>>
    where
        C: TreeSearchCost + Display + Send + Clone,
    {
        let mut max_cost_info = None;
        let mut max_cost = f64::MIN;
        for child_idx in &cost.tree().node(node_idx).children {
            // TODO: parallelization?
            let move_cost_info =
                calc_nni_cost_with_blen_opt(node_idx, child_idx, base_cost, cost.clone())?;
            if move_cost_info.cost() > max_cost {
                max_cost = move_cost_info.cost();
                max_cost_info = Some(move_cost_info);
            }
        }
        Ok(max_cost_info)
    }
}

impl Display for NniOptimiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NNI")
    }
}

fn calc_nni_cost_with_blen_opt<C: TreeSearchCost + Clone + Display>(
    node_idx: &NodeIdx,
    child_idx: &NodeIdx,
    base_cost: f64,
    mut cost_fn: C,
) -> Result<MoveCostInfo> {
    let mut new_tree = rooted_nni(cost_fn.tree(), node_idx, child_idx)?;
    cost_fn.update_tree(new_tree.clone(), &[*node_idx]);
    let mut move_cost = cost_fn.cost();
    if cost_fn.blen_optimisation() && move_cost <= base_cost {
        let mut o = BranchOptimiser::new(cost_fn);
        let blen_opt = o.optimise_branch(node_idx)?;
        if blen_opt.final_cost > move_cost {
            move_cost = blen_opt.final_cost;
            new_tree.set_blen(node_idx, blen_opt.value);
        }
    }
    Ok(MoveCostInfo::new(move_cost, new_tree, vec![*node_idx]))
}

fn rooted_nni(tree: &Tree, node_idx: &NodeIdx, child_idx: &NodeIdx) -> Result<Tree> {
    if node_idx == &tree.root {
        bail!("For the rooted NNI the node mustn't be the root of the tree.");
    }
    if matches!(node_idx, Leaf(_)) {
        bail!("For the rooted NNI the node mustn't be a leaf");
    }
    if tree.node(child_idx).parent.is_none() || tree.node(child_idx).parent.unwrap() != *node_idx {
        bail!("The provided child_idx (i.e. the node that indicates which subtrees should be swapped) is not a child of the node node_idx.");
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
fn rooted_nni_unchecked(tree: &Tree, node_idx: &NodeIdx, child_idx: &NodeIdx) -> Tree {
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

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
pub mod private_nni_tests {

    use crate::tree;
    use crate::tree::Tree;

    use super::*;

    #[cfg(test)]
    fn compare_trees(tree: &Tree, true_tree: Tree) {
        assert_eq!(tree.root, true_tree.root);
        for node_idx in tree.preorder() {
            let current = tree.node(node_idx);
            let current_id = &current.id;
            assert_eq!(current.blen, true_tree.by_id(current_id).blen);
            if node_idx == &tree.root {
                continue;
            }
            let true_parent = true_tree.by_id(current_id);
            let parent = tree.by_id(current_id);
            assert_eq!(parent.id, true_parent.id);
        }
    }
    #[test]
    fn nni_in_middle_of_tree() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let true_tree_after_nni =
            tree!("(((A:1.0,B:1.0)F:1.0,(D:3.0,C:2.0)G:1.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "G";
        let child_id = "F";

        // act
        let new_tree =
            rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap();

        // assert
        compare_trees(&new_tree, true_tree_after_nni);
        let dirty_nodes: Vec<_> = tree
            .postorder()
            .iter()
            .filter(|&x| new_tree.dirty[usize::from(x)])
            .collect();
        assert_eq!(dirty_nodes.len(), 1);
        assert_eq!(tree.node(dirty_nodes.first().unwrap()).id, node_id);
    }

    #[test]
    fn nni_at_parent_of_leaf() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let true_tree_after_nni =
            tree!("((((C:2.0,B:1.0)F:1.0,A:1.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "F";
        let child_id = "A";

        // act
        let new_tree =
            rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap();

        // assert
        compare_trees(&new_tree, true_tree_after_nni);
        let dirty_nodes: Vec<_> = tree
            .postorder()
            .iter()
            .filter(|&x| new_tree.dirty[usize::from(x)])
            .collect();
        assert_eq!(dirty_nodes.len(), 1);
        assert_eq!(tree.node(dirty_nodes.first().unwrap()).id, node_id);
    }

    #[test]
    fn nni_node_is_root() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "I";

        // act
        let err = rooted_nni(&tree, &tree.by_id(node_id).idx, &Leaf(0)).unwrap_err();

        // assert
        assert!(err.to_string().contains("root"));
    }

    #[test]
    fn nni_node_is_leaf() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "A";

        // act
        let err = rooted_nni(&tree, &tree.by_id(node_id).idx, &Leaf(0)).unwrap_err();

        // assert
        assert!(err.to_string().contains("leaf"));
    }
    #[test]
    fn nni_child_is_invalid() {
        // arrange
        let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
        let node_id = "G";
        let child_id = "A";

        // act
        let err =
            rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap_err();

        // assert
        assert!(err.to_string().contains("child"));
    }
}
