use super::njmat;
use super::Result;

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) idx: usize,
    pub(crate) parent: Option<usize>,
    pub(crate) children: Vec<usize>,
    pub(crate) blen: f32,
}

impl Node {
    pub(crate) fn add_parent(&mut self, parent_idx: usize, blen: f32) {
        self.parent = Some(parent_idx);
        self.blen = blen;
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct Tree {
    pub(crate) root: isize,
    pub(crate) nodes: Vec<Node>,
    pub(crate) postorder: Vec<usize>,
    leaf_number: usize,
}

impl Tree {
    pub(crate) fn new(n: usize, root: usize) -> Self {
        Self {
            root: root as isize,
            nodes: (0..n)
                .map(|idx| Node {
                    idx: idx,
                    parent: None,
                    children: Vec::new(),
                    blen: 0.0,
                })
                .collect(),
            postorder: Vec::new(),
            leaf_number: n,
        }
    }
    pub(crate) fn add_parent(
        &mut self,
        idx_new: usize,
        idx_i: usize,
        idx_j: usize,
        blen_i: f32,
        blen_j: f32,
    ) {
        self.nodes.push(Node {
            idx: idx_new,
            parent: None,
            children: vec![idx_i, idx_j],
            blen: 0.0,
        });
        self.nodes[idx_i].add_parent(idx_new, blen_i);
        self.nodes[idx_j].add_parent(idx_new, blen_j);
    }

    pub(crate) fn create_postorder(&mut self) {
        if self.postorder.len() == 0 {
                let mut order = Vec::<usize>::with_capacity(self.nodes.len());
                let mut stack = Vec::<usize>::with_capacity(self.nodes.len());
                let mut cur_root = self.root as usize;
                stack.push(cur_root);
                while !stack.is_empty() {
                    cur_root = stack.pop().unwrap();
                    order.push(cur_root);
                    if !self.is_leaf(cur_root) {
                        stack.push(self.nodes[cur_root].children[0]);
                        stack.push(self.nodes[cur_root].children[1]);
                    }
                }
                order.reverse();
                self.postorder = order;
        }
    }

    pub(crate) fn is_leaf(&self, node_idx: usize) -> bool {
        node_idx < self.leaf_number
    }
}

pub(crate) fn build_nj_tree(mut nj_data: njmat::NJMat) -> Result<Tree> {
    let n = nj_data.distances.ncols();
    let root_idx = n * 2 - 2;
    let mut tree = Tree::new(n, root_idx);

    for cur_idx in n..=root_idx {
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
    Ok(tree)
}

pub(crate) fn traverse_tree(tree: &mut Tree) {
    let order = tree.create_postorder(); // should not be mutable
}
