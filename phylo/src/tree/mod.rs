use std::fmt::{Debug, Display};

use anyhow::bail;
use approx::relative_eq;
use bio::alignment::distance::levenshtein;
use inc_stats::Percentiles;
use nalgebra::{max, DMatrix};
use rand::random;

use crate::alignment::Sequences;
use crate::tree::{
    nj_matrices::{Mat, NJMat},
    NodeIdx::{Internal as Int, Leaf},
};
use crate::{Result, Rounding};

mod nj_matrices;
pub mod tree_parser;

mod tree_node;
pub use tree_node::*;

#[derive(PartialEq, Clone, Copy, PartialOrd, Eq, Ord, Hash)]
pub enum NodeIdx {
    Internal(usize),
    Leaf(usize),
}

impl Display for NodeIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Int(idx) => write!(f, "internal node {}", idx),
            Leaf(idx) => write!(f, "leaf node {}", idx),
        }
    }
}

impl Debug for NodeIdx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Int(idx) => write!(f, "Int({})", idx),
            Leaf(idx) => write!(f, "Leaf({})", idx),
        }
    }
}

impl From<&NodeIdx> for usize {
    fn from(node_idx: &NodeIdx) -> usize {
        Self::from(*node_idx)
    }
}

impl From<NodeIdx> for usize {
    fn from(node_idx: NodeIdx) -> usize {
        match node_idx {
            Int(idx) => idx,
            Leaf(idx) => idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tree {
    pub root: NodeIdx,
    pub(crate) nodes: Vec<Node>,
    postorder: Vec<NodeIdx>,
    preorder: Vec<NodeIdx>,
    pub complete: bool,
    pub n: usize,
    pub height: f64,
}

impl Tree {
    pub fn new(sequences: &Sequences) -> Result<Self> {
        let n = sequences.len();
        if n == 0 {
            bail!("No sequences provided, aborting.");
        }
        if n == 1 {
            Ok(Self {
                root: Leaf(0),
                postorder: vec![Leaf(0)],
                preorder: vec![Leaf(0)],
                nodes: vec![Node::new_leaf(
                    0,
                    None,
                    0.0,
                    sequences.record(0).id().to_string(),
                )],
                complete: true,
                n: 1,
                height: 0.0,
            })
        } else {
            Ok(Self {
                root: Int(2 * n - 2),
                postorder: Vec::new(),
                preorder: Vec::new(),
                nodes: (0..n)
                    .zip(sequences.iter().map(|seq| seq.id().to_string()))
                    .map(|(idx, id)| Node::new_leaf(idx, None, 0.0, id))
                    .collect(),
                complete: false,
                n,
                height: 0.0,
            })
        }
    }

    fn is_subtree(&self, query: &NodeIdx, node: &NodeIdx) -> bool {
        let order = self.preorder_subroot(node);
        order.contains(query)
    }

    pub fn sibling(&self, node_idx: &NodeIdx) -> Option<NodeIdx> {
        let parent = self.nodes[usize::from(node_idx)].parent?;
        let siblings = &self.nodes[usize::from(&parent)].children;
        siblings
            .iter()
            .find(|&sibling| sibling != node_idx)
            .cloned()
    }

    pub fn rooted_spr(&self, prune_idx: &NodeIdx, regraft_idx: &NodeIdx) -> Result<Tree> {
        // Prune and regraft nodes must be different
        if prune_idx == regraft_idx {
            bail!("Prune and regraft nodes must be different.");
        }
        if self.is_subtree(regraft_idx, prune_idx) {
            bail!("Prune node cannot be a subtree of the regraft node.");
        }

        let prune = self.node(prune_idx);
        // Pruned node must have a parent, it is the one being reattached
        if prune.parent.is_none() {
            bail!("Cannot prune the root node.");
        }
        // Cannot prune direct child of the root node, otherwise branch lengths are undefined
        if self.node(&prune.parent.unwrap()).parent.is_none() {
            bail!("Cannot prune direct child of the root node.");
        }
        let regraft = self.node(regraft_idx);
        // Regrafted node must have a parent, the prune parent is attached to that branch
        if regraft.parent.is_none() {
            bail!("Cannot regraft to root node.");
        }
        if regraft.parent == prune.parent {
            bail!("Prune and regraft nodes must have different parents.");
        }

        Ok(self.rooted_spr_unchecked(prune_idx, regraft_idx))
    }

    fn rooted_spr_unchecked(&self, prune_idx: &NodeIdx, regraft_idx: &NodeIdx) -> Tree {
        let prune = self.node(prune_idx);
        let prune_sib = self.node(&self.sibling(&prune.idx).unwrap());
        let prune_par = self.node(&prune.parent.unwrap());
        let prune_grpar = self.node(&prune_par.parent.unwrap());
        let regraft = self.node(regraft_idx);
        let regraft_par = self.node(&regraft.parent.unwrap());

        let mut new_tree = self.clone();
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
        debug_assert_eq!(new_tree.postorder.len(), self.postorder.len());
        new_tree
    }

    pub fn iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    pub fn children(&self, node_idx: &NodeIdx) -> &Vec<NodeIdx> {
        &self.nodes[usize::from(node_idx)].children
    }

    pub fn node(&self, node_idx: &NodeIdx) -> &Node {
        &self.nodes[usize::from(node_idx)]
    }

    pub fn node_mut(&mut self, node_idx: &NodeIdx) -> &mut Node {
        &mut self.nodes[usize::from(node_idx)]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn preorder(&self) -> &Vec<NodeIdx> {
        &self.preorder
    }

    pub fn postorder(&self) -> &Vec<NodeIdx> {
        &self.postorder
    }

    pub fn to_newick(&self) -> String {
        format!("({});", self.to_newick_subroot(self.root))
    }

    fn to_newick_subroot(&self, node_idx: NodeIdx) -> String {
        match node_idx {
            NodeIdx::Leaf(idx) => {
                let node = &self.nodes[idx];
                format!("{}:{}", node.id, node.blen)
            }
            NodeIdx::Internal(idx) => {
                let node = &self.nodes[idx];
                let children_newick: Vec<String> = node
                    .children
                    .iter()
                    .map(|&child_idx| self.to_newick_subroot(child_idx))
                    .collect();
                format!("({}){}:{}", children_newick.join(","), &node.id, node.blen)
            }
        }
    }

    pub fn node_id(&self, node_idx: &NodeIdx) -> &str {
        &self.nodes[usize::from(node_idx)].id
    }

    pub(crate) fn add_parent(
        &mut self,
        parent_idx: usize,
        idx_i: &NodeIdx,
        idx_j: &NodeIdx,
        blen_i: f64,
        blen_j: f64,
    ) {
        self.nodes.push(Node::new_internal(
            parent_idx,
            None,
            vec![*idx_i, *idx_j],
            0.0,
            "".to_string(),
        ));
        self.add_parent_to_child(idx_i, &Int(parent_idx), blen_i);
        self.add_parent_to_child(idx_j, &Int(parent_idx), blen_j);
    }

    pub(crate) fn add_parent_to_child(&mut self, idx: &NodeIdx, parent_idx: &NodeIdx, blen: f64) {
        self.nodes[usize::from(idx)].add_parent(parent_idx);
        self.nodes[usize::from(idx)].blen = blen;
    }

    pub(crate) fn add_parent_to_child_no_blen(&mut self, idx: &NodeIdx, parent_idx: &NodeIdx) {
        self.nodes[usize::from(idx)].add_parent(parent_idx);
    }

    pub(crate) fn compute_postorder(&mut self) {
        debug_assert!(self.complete);
        let mut order = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut stack = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut cur_root = self.root;
        stack.push(cur_root);
        while !stack.is_empty() {
            cur_root = stack.pop().unwrap();
            order.push(cur_root);
            if let Int(idx) = cur_root {
                stack.push(self.nodes[idx].children[0]);
                stack.push(self.nodes[idx].children[1]);
            }
        }
        order.reverse();
        self.postorder = order;
    }

    pub(crate) fn compute_preorder(&mut self) {
        debug_assert!(self.complete);
        self.preorder = self.preorder_subroot(&self.root);
    }

    pub fn preorder_subroot(&self, subroot_idx: &NodeIdx) -> Vec<NodeIdx> {
        debug_assert!(self.complete);
        let mut order = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut stack = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut cur_root = *subroot_idx;
        stack.push(cur_root);
        while !stack.is_empty() {
            cur_root = stack.pop().unwrap();
            order.push(cur_root);
            if let Int(idx) = cur_root {
                for child in self.nodes[idx].children.iter().rev() {
                    stack.push(*child);
                }
            }
        }
        order
    }

    pub fn leaf_ids(&self) -> Vec<String> {
        debug_assert!(self.complete);
        self.leaves().iter().map(|node| node.id.clone()).collect()
    }

    pub fn try_idx(&self, id: &str) -> Result<NodeIdx> {
        debug_assert!(self.complete);
        let node = self.nodes.iter().find(|node| node.id == id);
        if let Some(node) = node {
            return Ok(node.idx);
        }
        bail!("No node with id {} found in the tree", id);
    }

    pub fn idx(&self, id: &str) -> NodeIdx {
        debug_assert!(self.complete);
        self.nodes.iter().find(|node| node.id == id).unwrap().idx
    }

    pub fn set_blen(&mut self, node_idx: &NodeIdx, blen: f64) {
        debug_assert!(blen >= 0.0);
        let old_blen = self.nodes[usize::from(node_idx)].blen;
        self.height += blen - old_blen;
        self.nodes[usize::from(node_idx)].blen = blen;
    }

    pub fn blen(&self, node_idx: &NodeIdx) -> f64 {
        self.nodes[usize::from(node_idx)].blen
    }

    pub fn leaves(&self) -> Vec<&Node> {
        debug_assert!(self.complete);
        self.nodes
            .iter()
            .filter(|&x| matches!(x.idx, Leaf(_)))
            .collect()
    }

    pub fn internals(&self) -> Vec<&Node> {
        debug_assert!(self.complete);
        self.nodes
            .iter()
            .filter(|&x| matches!(x.idx, Int(_)))
            .collect()
    }
}

pub fn percentiles(lengths: &[f64], categories: u32) -> Vec<f64> {
    percentiles_rounded(lengths, categories, &Rounding::none())
}

pub fn percentiles_rounded(lengths: &[f64], categories: u32, rounding: &Rounding) -> Vec<f64> {
    let lengths: Percentiles<f64> = lengths.iter().collect();
    let percentiles: Vec<f64> = (1..(categories + 1))
        .map(|cat| 1.0 / ((categories + 1) as f64) * (cat as f64))
        .collect();
    let mut values = lengths.percentiles(percentiles).unwrap().unwrap();
    if rounding.round {
        values.iter_mut().for_each(|len| {
            *len = (*len * (10.0_f64.powf(rounding.digits as f64))).round()
                / (10.0_f64.powf(rounding.digits as f64))
        });
    }
    values
}

// #[allow(dead_code)]
fn argmin_wo_diagonal(q: Mat, rng: fn(usize) -> usize) -> (usize, usize) {
    debug_assert!(!q.is_empty(), "The input matrix must not be empty.");
    debug_assert!(
        q.ncols() > 1 && q.nrows() > 1,
        "The input matrix should have more than 1 element."
    );
    let mut arg_min = vec![];
    let mut val_min = &f64::MAX;
    for i in 0..q.nrows() {
        for j in 0..i {
            let val = &q[(i, j)];
            if val < val_min {
                val_min = val;
                arg_min = vec![(i, j)];
            } else if val == val_min {
                arg_min.push((i, j));
            }
        }
    }
    arg_min[(rng)(arg_min.len())]
}

fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}

fn build_nj_tree_w_rng_from_matrix(
    mut nj_data: NJMat,
    sequences: &Sequences,
    rng: fn(usize) -> usize,
) -> Result<Tree> {
    let n = nj_data.distances.ncols();
    let mut tree = Tree::new(sequences)?;
    let root_idx = usize::from(&tree.root);
    for cur_idx in n..=root_idx {
        let q = nj_data.compute_nj_q();
        let (i, j) = argmin_wo_diagonal(q, rng);
        let idx_new = cur_idx;
        let (blen_i, blen_j) = nj_data.branch_lengths(i, j, cur_idx == root_idx);
        tree.add_parent(idx_new, &nj_data.idx[i], &nj_data.idx[j], blen_i, blen_j);
        nj_data = nj_data
            .add_merge_node(idx_new)
            .recompute_new_node_distances(i, j)
            .remove_merged_nodes(i, j);
    }
    tree.n = n;
    tree.complete = true;
    tree.compute_postorder();
    tree.compute_preorder();
    tree.height = tree.nodes.iter().map(|node| node.blen).sum();
    Ok(tree)
}

fn build_nj_tree_from_matrix(nj_data: NJMat, sequences: &Sequences) -> Result<Tree> {
    build_nj_tree_w_rng_from_matrix(nj_data, sequences, rng_len)
}

pub fn build_nj_tree(sequences: &Sequences) -> Result<Tree> {
    let nj_data = compute_distance_matrix(sequences);
    build_nj_tree_from_matrix(nj_data, sequences)
}

fn compute_distance_matrix(sequences: &Sequences) -> nj_matrices::NJMat {
    let nseqs = sequences.len();
    let mut distances = DMatrix::zeros(nseqs, nseqs);
    for i in 0..nseqs {
        for j in (i + 1)..nseqs {
            let seq_i = sequences.record(i).seq();
            let seq_j = sequences.record(j).seq();
            let lev_dist = levenshtein(seq_i, seq_j) as f64;
            let proportion_diff = f64::min(
                lev_dist / (max(seq_i.len(), seq_j.len()) as f64),
                0.75 - f64::EPSILON,
            );
            let corrected_dist = -3.0 / 4.0 * (1.0 - 4.0 / 3.0 * proportion_diff).ln();
            distances[(i, j)] = corrected_dist;
            distances[(j, i)] = corrected_dist;
        }
    }
    nj_matrices::NJMat {
        idx: (0..nseqs).map(NodeIdx::Leaf).collect(),
        distances,
    }
}

#[cfg(test)]
mod tests;
