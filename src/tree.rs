use super::njmat::NJMat;
use super::Result;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord)]
pub(crate) enum NodeIdx {
    Internal(usize),
    Leaf(usize),
}

use NodeIdx::Internal as Int;
use NodeIdx::Leaf;

impl Into<usize> for NodeIdx {
    fn into(self) -> usize {
        match self {
            Int(idx) => idx,
            Leaf(idx) => idx,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) idx: usize,
    pub(crate) parent: Option<NodeIdx>,
    pub(crate) children: Vec<NodeIdx>,
    pub(crate) blen: f32,
    pub(crate) id: String,
}

impl Node {
    fn new_empty(node_idx: usize) -> Self {
        Self::new_with_children(node_idx, Vec::new())
    }

    fn new_with_children(node_idx: usize, children: Vec<NodeIdx>) -> Self {
        Self {
            idx: node_idx,
            parent: None,
            children: children,
            blen: 0.0,
            id: String::from(""),
        }
    }

    pub(crate) fn add_parent(&mut self, parent_idx: NodeIdx, blen: f32) {
        assert!(matches!(parent_idx, Int(_)));
        self.parent = Some(parent_idx);
        self.blen = blen;
    }
}

#[derive(Debug)]
pub(crate) struct Tree {
    pub(crate) root: NodeIdx,
    pub(crate) postorder: Vec<NodeIdx>,
    pub(crate) preorder: Vec<NodeIdx>,
    pub(crate) leaf_number: usize,
    pub(crate) leaves: Vec<Node>,
    pub(crate) internals: Vec<Node>,
}

impl Tree {
    pub(crate) fn new(n: usize, root: usize) -> Self {
        Self {
            root: Int(root),
            postorder: Vec::new(),
            preorder: Vec::new(),
            leaf_number: n,
            leaves: (0..n).map(Node::new_empty).collect(),
            internals: Vec::new(),
        }
    }

    pub(crate) fn add_parent(
        &mut self,
        parent_idx: usize,
        idx_i: NodeIdx,
        idx_j: NodeIdx,
        blen_i: f32,
        blen_j: f32,
    ) {
        self.internals
            .push(Node::new_with_children(parent_idx, vec![idx_i, idx_j]));
        self.add_parent_to_child(&idx_i, parent_idx, blen_i);
        self.add_parent_to_child(&idx_j, parent_idx, blen_j);
    }

    fn add_parent_to_child(&mut self, idx: &NodeIdx, parent_idx: usize, blen: f32) {
        match *idx {
            Int(idx) => self.internals[idx].add_parent(Int(parent_idx), blen),
            Leaf(idx) => self.leaves[idx].add_parent(Int(parent_idx), blen),
        }
    }

    pub(crate) fn create_postorder(&mut self) {
        if self.postorder.len() == 0 {
            let mut order = Vec::<NodeIdx>::with_capacity(self.leaves.len() + self.internals.len());
            let mut stack = Vec::<NodeIdx>::with_capacity(self.internals.len());
            let mut cur_root = self.root;
            stack.push(cur_root.clone());
            while !stack.is_empty() {
                cur_root = stack.pop().unwrap();
                order.push(cur_root.clone());
                if let Int(idx) = cur_root {
                    stack.push(self.internals[idx].children[0]);
                    stack.push(self.internals[idx].children[1]);
                }
            }
            order.reverse();
            self.postorder = order;
        }
    }

    pub(crate) fn create_preorder(&mut self) {
        if self.preorder.len() == 0 {
            self.preorder = self.preorder_subroot(self.root);
        }
    }

    pub(crate) fn preorder(&self) -> Vec<NodeIdx> {
        self.preorder.clone()
    }

    pub(crate) fn preorder_subroot(&self, subroot_idx: NodeIdx) -> Vec<NodeIdx> {
        let mut order = Vec::<NodeIdx>::with_capacity(self.leaves.len() + self.internals.len());
        let mut stack = Vec::<NodeIdx>::with_capacity(self.internals.len());
        let mut cur_root = subroot_idx;
        stack.push(cur_root);
        while !stack.is_empty() {
            cur_root = stack.pop().unwrap();
            order.push(cur_root);
            if let Int(idx) = cur_root {
                for child in self.internals[idx].children.iter().rev() {
                    stack.push(*child);
                }
            }
        }
        order
    }
}

pub(crate) fn build_nj_tree(mut nj_data: NJMat) -> Result<Tree> {
    let n = nj_data.distances.ncols();
    let root_idx = n - 2;
    let mut tree = Tree::new(n, root_idx);

    for cur_idx in 0..=root_idx {
        let q = nj_data.compute_nj_q();
        let (i, j) = q.iamax_full();
        let idx_new = cur_idx;

        let (blen_i, blen_j) = nj_data.branch_lengths(i, j, cur_idx == root_idx);

        tree.add_parent(idx_new, nj_data.idx[i], nj_data.idx[j], blen_i, blen_j);

        nj_data = nj_data
            .add_merge_node(idx_new)
            .recompute_new_node_distances(i, j)
            .remove_merged_nodes(i, j);
    }
    tree.create_postorder();
    tree.create_preorder();
    Ok(tree)
}

#[cfg(test)]
mod tree_tests {
    use super::super::njmat::NJMat;
    use super::{build_nj_tree, Node, NodeIdx, NodeIdx::Internal as I, NodeIdx::Leaf as L, Tree};
    use approx::relative_eq;
    use nalgebra::dmatrix;

    #[cfg(test)]
    pub(crate) fn setup_test_tree() -> Tree {
        let mut tree = Tree::new(5, 3);
        tree.add_parent(0, L(0), L(1), 1.0, 1.0);
        tree.add_parent(1, L(3), L(4), 1.0, 1.0);
        tree.add_parent(2, L(2), I(1), 1.0, 1.0);
        tree.add_parent(3, I(0), I(2), 1.0, 1.0);
        tree.create_postorder();
        tree.create_preorder();
        tree
    }

    #[test]
    pub(crate) fn subroot_preorder() {
        let tree = setup_test_tree();
        assert_eq!(tree.preorder_subroot(I(0)), [I(0), L(0), L(1)]);
        assert_eq!(tree.preorder_subroot(I(1)), [I(1), L(3), L(4)]);
        assert_eq!(tree.preorder_subroot(I(2)), [I(2), L(2), I(1), L(3), L(4)]);
        assert_eq!(
            tree.preorder_subroot(I(3)),
            [I(3), I(0), L(0), L(1), I(2), L(2), I(1), L(3), L(4)]
        );
        assert_eq!(tree.preorder_subroot(I(3)), tree.preorder());
    }

    #[test]
    pub(crate) fn postorder() {
        let tree = setup_test_tree();
        assert_eq!(
            tree.postorder,
            [L(0), L(1), I(0), L(2), L(3), L(4), I(1), I(2), I(3)]
        );
    }

    #[cfg(test)]
    impl PartialEq for Node {
        fn eq(&self, other: &Self) -> bool {
            (self.idx == other.idx)
                && (self.parent == other.parent)
                && (self.children.iter().min() == other.children.iter().min())
                && (self.children.iter().max() == other.children.iter().max())
                && relative_eq!(self.blen, other.blen)
        }
    }

    #[cfg(test)]
    impl Node {
        pub(crate) fn new_leaf(idx: usize, parent: Option<NodeIdx>, blen: f32, id: &str) -> Self {
            Self {
                idx,
                parent,
                children: Vec::new(),
                blen,
                id: id.to_string(),
            }
        }

        pub(crate) fn new_internal(
            idx: usize,
            parent: Option<NodeIdx>,
            children: Vec<NodeIdx>,
            blen: f32,
            id: &str,
        ) -> Self {
            Self {
                idx,
                parent,
                children,
                blen,
                id: id.to_string(),
            }
        }
    }

    #[test]
    pub(crate) fn nj_correct() {
        let nj_distances = NJMat {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let nj_tree = build_nj_tree(nj_distances).unwrap();
        let leaves = vec![
            Node::new_leaf(0, Some(I(0)), 2.0, ""),
            Node::new_leaf(1, Some(I(0)), 3.0, ""),
            Node::new_leaf(2, Some(I(1)), 4.0, ""),
            Node::new_leaf(3, Some(I(2)), 2.0, ""),
            Node::new_leaf(4, Some(I(2)), 1.0, ""),
        ];
        let internals = vec![
            Node::new_internal(0, Some(I(1)), vec![L(1), L(0)], 3.0, ""),
            Node::new_internal(1, Some(I(3)), vec![L(2), I(0)], 1.0, ""),
            Node::new_internal(2, Some(I(3)), vec![L(4), L(3)], 1.0, ""),
            Node::new_internal(3, None, vec![I(2), I(1)], 0.0, ""),
        ];
        assert_eq!(nj_tree.root, I(3));
        for (i, node) in nj_tree.leaves.into_iter().enumerate() {
            assert_eq!(node, leaves[i]);
        }
        for (i, node) in nj_tree.internals.into_iter().enumerate() {
            assert_eq!(node, internals[i]);
        }
    }
}
