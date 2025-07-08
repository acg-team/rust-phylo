use itertools::Itertools;

use crate::tree;
use crate::tree::{NodeIdx::Leaf, Tree};

use super::rooted_nni;

#[cfg(test)]
fn compare_trees(tree: &Tree, true_tree: Tree) {
    assert_eq!(tree.root, true_tree.root);
    for node_idx in tree.preorder() {
        let current = tree.node(node_idx);
        let current_id = current.id.clone();
        assert_eq!(current.blen, true_tree.by_id(&current_id).blen);
        if node_idx == &tree.root {
            continue;
        }
        let true_parent = true_tree.by_id(&current_id);
        let parent = tree.by_id(&current_id);
        assert_eq!(parent.id, true_parent.id);
    }
}
#[test]
fn nni_in_middle_of_tree() {
    // arrange
    let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
    let true_tree_after_nni = tree!("(((A:1.0,B:1.0)F:1.0,(D:3.0,C:2.0)G:1.0)H:1.0,E:4.0)I:1.0;");
    let node_id = "G";
    let child_id = "F";

    // act
    let new_tree = rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap();

    // assert
    compare_trees(&new_tree, true_tree_after_nni);
    let dirty_nodes = tree
        .postorder()
        .iter()
        .filter(|&x| new_tree.dirty[usize::from(x)])
        .collect_vec();
    assert_eq!(dirty_nodes.len(), 1);
    assert_eq!(tree.node(dirty_nodes.first().unwrap()).id, node_id);
}

#[test]
fn nni_at_parent_of_leaf() {
    // arrange
    let tree = tree!("((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
    let true_tree_after_nni = tree!("((((C:2.0,B:1.0)F:1.0,A:1.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;");
    let node_id = "F";
    let child_id = "A";

    // act
    let new_tree = rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap();

    // assert
    compare_trees(&new_tree, true_tree_after_nni);
    let dirty_nodes = tree
        .postorder()
        .iter()
        .filter(|&x| new_tree.dirty[usize::from(x)])
        .collect_vec();
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
    let err = rooted_nni(&tree, &tree.by_id(node_id).idx, &tree.by_id(child_id).idx).unwrap_err();

    // assert
    assert!(err.to_string().contains("child"));
}
