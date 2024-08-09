use std::fmt::Display;

use anyhow::bail;
use approx::relative_eq;
use bio::alignment::distance::levenshtein;
use bio::io::fasta::Record;
use inc_stats::Percentiles;
use log::info;
use nalgebra::{max, DMatrix};
use rand::random;

use crate::tree::{
    nj_matrices::{Mat, NJMat},
    NodeIdx::{Internal as Int, Leaf},
};
use crate::{Result, Rounding};

mod nj_matrices;
pub mod tree_parser;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord, Hash)]
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

impl From<&NodeIdx> for usize {
    fn from(node_idx: &NodeIdx) -> usize {
        match node_idx {
            Int(idx) => *idx,
            Leaf(idx) => *idx,
        }
    }
}

impl From<NodeIdx> for usize {
    fn from(node_idx: NodeIdx) -> usize {
        Self::from(&node_idx)
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub idx: NodeIdx,
    pub parent: Option<NodeIdx>,
    pub children: Vec<NodeIdx>,
    pub blen: f64,
    pub id: String,
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.idx {
            Int(idx) => write!(f, "Internal node {}", idx),
            Leaf(idx) => write!(f, "Leaf node {}", idx),
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        (self.idx == other.idx)
            && (self.parent == other.parent)
            && (self.children.iter().min() == other.children.iter().min())
            && (self.children.iter().max() == other.children.iter().max())
            && relative_eq!(self.blen, other.blen)
    }
}

impl Node {
    fn new_leaf(idx: usize, parent: Option<NodeIdx>, blen: f64, id: String) -> Self {
        Self {
            idx: Leaf(idx),
            parent,
            children: Vec::new(),
            blen,
            id,
        }
    }

    fn new_internal(
        idx: usize,
        parent: Option<NodeIdx>,
        children: Vec<NodeIdx>,
        blen: f64,
        id: String,
    ) -> Self {
        Self {
            idx: Int(idx),
            parent,
            children,
            blen,
            id,
        }
    }

    fn new_empty_internal(node_idx: usize) -> Self {
        Self::new_internal(node_idx, None, Vec::new(), 0.0, "".to_string())
    }

    fn add_parent(&mut self, parent_idx: &NodeIdx) {
        debug_assert!(matches!(parent_idx, Int(_)));
        self.parent = Some(*parent_idx);
    }
}

#[derive(Debug, Clone)]
pub struct Tree {
    pub root: NodeIdx,
    pub nodes: Vec<Node>,
    pub postorder: Vec<NodeIdx>,
    pub preorder: Vec<NodeIdx>,
    pub complete: bool,
    pub n: usize,
}

impl Tree {
    pub fn new(sequences: &[Record]) -> Result<Self> {
        let n = sequences.len();
        if n == 0 {
            bail!("No sequences provided, aborting.");
        }
        if n == 1 {
            Ok(Self {
                root: Leaf(0),
                postorder: vec![Leaf(0)],
                preorder: vec![Leaf(0)],
                nodes: vec![Node::new_leaf(0, None, 0.0, sequences[0].id().to_string())],
                complete: true,
                n: 1,
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
            })
        }
    }

    pub fn children(&self, node_idx: &NodeIdx) -> &Vec<NodeIdx> {
        &self.nodes[usize::from(node_idx)].children
    }

    pub fn to_newick(&self) -> String {
        format!("({});", self._to_newick(self.root))
    }

    fn _to_newick(&self, node_idx: NodeIdx) -> String {
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
                    .map(|&child_idx| self._to_newick(child_idx))
                    .collect();
                format!("({}){}:{}", children_newick.join(","), &node.id, node.blen)
            }
        }
    }

    pub fn node_id(&self, node_idx: &NodeIdx) -> String {
        let id = &self.nodes[usize::from(node_idx)].id;
        if id.is_empty() {
            String::new()
        } else {
            format!(" with id {}", id)
        }
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

    pub(crate) fn create_postorder(&mut self) {
        debug_assert!(self.complete);
        if self.postorder.is_empty() {
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
    }

    pub(crate) fn create_preorder(&mut self) {
        debug_assert!(self.complete);
        if self.preorder.is_empty() {
            self.preorder = self.preorder_subroot(Some(self.root));
        }
    }

    pub fn preorder_subroot(&self, subroot_idx: Option<NodeIdx>) -> Vec<NodeIdx> {
        debug_assert!(self.complete);
        let subroot_idx = match subroot_idx {
            Some(idx) => idx,
            None => self.root,
        };
        let mut order = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut stack = Vec::<NodeIdx>::with_capacity(self.nodes.len());
        let mut cur_root = subroot_idx;
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

    pub fn all_branch_lengths(&self) -> Vec<f64> {
        debug_assert!(self.complete);
        let lengths = self.nodes.iter().map(|n| n.blen).collect();
        info!("Branch lengths are: {:?}", lengths);
        lengths
    }

    pub fn idx(&self, id: &str) -> Result<usize> {
        debug_assert!(self.complete);
        let idx = self.nodes.iter().position(|node| node.id == id);
        if let Some(idx) = idx {
            return Ok(idx);
        }
        bail!("No node with id {} found in the tree", id);
    }

    pub fn set_branch_length(&mut self, node_idx: &NodeIdx, blen: f64) {
        debug_assert!(blen >= 0.0);
        self.nodes[usize::from(node_idx)].blen = blen;
    }

    pub fn branch_length(&self, node_idx: &NodeIdx) -> f64 {
        self.nodes[usize::from(node_idx)].blen
    }

    pub fn leaves(&self) -> Vec<&Node> {
        debug_assert!(self.complete);
        self.nodes
            .iter()
            .filter(|&x| matches!(x.idx, Leaf(_)))
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
    sequences: &[Record],
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
    tree.create_postorder();
    tree.create_preorder();
    Ok(tree)
}

fn build_nj_tree_from_matrix(nj_data: NJMat, sequences: &[Record]) -> Result<Tree> {
    build_nj_tree_w_rng_from_matrix(nj_data, sequences, rng_len)
}

pub fn build_nj_tree(sequences: &[Record]) -> Result<Tree> {
    let nj_data = compute_distance_matrix(sequences);
    build_nj_tree_from_matrix(nj_data, sequences)
}

fn compute_distance_matrix(sequences: &[Record]) -> nj_matrices::NJMat {
    let nseqs = sequences.len();
    let mut distances = DMatrix::zeros(nseqs, nseqs);
    for i in 0..nseqs {
        for j in (i + 1)..nseqs {
            let lev_dist = levenshtein(sequences[i].seq(), sequences[j].seq()) as f64;
            let proportion_diff = f64::min(
                lev_dist / (max(sequences[i].seq().len(), sequences[j].seq().len()) as f64),
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
mod tree_tests;
