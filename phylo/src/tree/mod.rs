use crate::tree::nj_matrices::{Mat, NJMat};
use crate::Result;
use crate::Rounding;
use bio::alignment::distance::levenshtein;
use bio::io::fasta::Record;
use inc_stats::Percentiles;
use log::info;
use nalgebra::{max, DMatrix};
use rand::random;
use std::fmt::Display;
use NodeIdx::{Internal as Int, Leaf};

mod nj_matrices;
pub(crate) mod tree_parser;

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

impl From<NodeIdx> for usize {
    fn from(node_idx: NodeIdx) -> usize {
        match node_idx {
            Int(idx) => idx,
            Leaf(idx) => idx,
        }
    }
}

#[derive(Debug)]
pub struct Node {
    #[allow(dead_code)]
    pub idx: NodeIdx,
    pub parent: Option<NodeIdx>,
    pub children: Vec<NodeIdx>,
    pub blen: f64,
    pub id: String,
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

    fn add_parent(&mut self, parent_idx: NodeIdx, blen: f64) {
        debug_assert!(matches!(parent_idx, Int(_)));
        self.parent = Some(parent_idx);
        self.blen = blen;
    }
}

#[derive(Debug)]
pub struct Tree {
    pub root: NodeIdx,
    pub leaves: Vec<Node>,
    pub internals: Vec<Node>,
    pub postorder: Vec<NodeIdx>,
    pub preorder: Vec<NodeIdx>,
    pub complete: bool,
}

impl Tree {
    pub fn new(sequences: &Vec<Record>) -> Self {
        let n = sequences.len();
        Self {
            root: Int(n - 2),
            postorder: Vec::new(),
            preorder: Vec::new(),
            leaves: (0..n)
                .zip(sequences.iter().map(|seq| seq.id().to_string()))
                .map(|(idx, id)| Node::new_leaf(idx, None, 0.0, id))
                .collect(),
            internals: Vec::with_capacity(n - 1),
            complete: false,
        }
    }

    pub fn get_node_id_string(&self, node_idx: &NodeIdx) -> String {
        let id = match node_idx {
            Int(idx) => &self.internals[*idx].id,
            Leaf(idx) => &self.leaves[*idx].id,
        };
        if id.is_empty() {
            String::new()
        } else {
            format!(" with id {}", id)
        }
    }

    pub fn add_parent(
        &mut self,
        parent_idx: usize,
        idx_i: NodeIdx,
        idx_j: NodeIdx,
        blen_i: f64,
        blen_j: f64,
    ) {
        self.internals.push(Node::new_internal(
            parent_idx,
            None,
            vec![idx_i, idx_j],
            0.0,
            "".to_string(),
        ));
        self.add_parent_to_child(&idx_i, parent_idx, blen_i);
        self.add_parent_to_child(&idx_j, parent_idx, blen_j);
    }

    pub fn add_parent_to_child(&mut self, idx: &NodeIdx, parent_idx: usize, blen: f64) {
        match *idx {
            Int(idx) => self.internals[idx].add_parent(Int(parent_idx), blen),
            Leaf(idx) => self.leaves[idx].add_parent(Int(parent_idx), blen),
        }
    }

    pub fn create_postorder(&mut self) {
        debug_assert!(self.complete);
        if self.postorder.is_empty() {
            let mut order = Vec::<NodeIdx>::with_capacity(self.leaves.len() + self.internals.len());
            let mut stack = Vec::<NodeIdx>::with_capacity(self.internals.len());
            let mut cur_root = self.root;
            stack.push(cur_root);
            while !stack.is_empty() {
                cur_root = stack.pop().unwrap();
                order.push(cur_root);
                if let Int(idx) = cur_root {
                    stack.push(self.internals[idx].children[0]);
                    stack.push(self.internals[idx].children[1]);
                }
            }
            order.reverse();
            self.postorder = order;
        }
    }

    pub fn create_preorder(&mut self) {
        debug_assert!(self.complete);
        if self.preorder.is_empty() {
            self.preorder = self.preorder_subroot(self.root);
        }
    }

    pub fn preorder_subroot(&self, subroot_idx: NodeIdx) -> Vec<NodeIdx> {
        debug_assert!(self.complete);
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

    pub fn get_leaf_ids(&self) -> Vec<String> {
        debug_assert!(self.complete);
        self.leaves.iter().map(|node| node.id.clone()).collect()
    }

    pub fn get_all_branch_lengths(&self) -> Vec<f64> {
        debug_assert!(self.complete);
        let lengths = self
            .leaves
            .iter()
            .map(|n| n.blen)
            .chain(self.internals.iter().map(|n| n.blen))
            .collect();
        info!("Branch lengths are: {:?}", lengths);
        lengths
    }
}

pub fn get_percentiles(lengths: &[f64], categories: u32) -> Vec<f64> {
    get_percentiles_rounded(lengths, categories, &Rounding::none())
}

pub fn get_percentiles_rounded(lengths: &[f64], categories: u32, rounding: &Rounding) -> Vec<f64> {
    let lengths: Percentiles<f64> = lengths.iter().collect();
    let percentiles: Vec<f64> = (1..(categories + 1))
        .map(|cat| 1.0 / ((categories + 1) as f64) * (cat as f64))
        .collect();
    let values = lengths.percentiles(percentiles).unwrap().unwrap();
    if rounding.round {
        values
            .iter()
            .map(|len| {
                (len * (10.0_f64.powf(rounding.digits as f64))).round()
                    / (10.0_f64.powf(rounding.digits as f64))
            })
            .collect()
    } else {
        values
    }
}

#[allow(dead_code)]
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
    sequences: &Vec<Record>,
    rng: fn(usize) -> usize,
) -> Result<Tree> {
    let n = nj_data.distances.ncols();
    let root_idx = n - 2;
    let mut tree = Tree::new(sequences);
    for cur_idx in 0..=root_idx {
        let q = nj_data.compute_nj_q();
        let (i, j) = argmin_wo_diagonal(q, rng);
        let idx_new = cur_idx;
        let (blen_i, blen_j) = nj_data.branch_lengths(i, j, cur_idx == root_idx);
        tree.add_parent(idx_new, nj_data.idx[i], nj_data.idx[j], blen_i, blen_j);
        nj_data = nj_data
            .add_merge_node(idx_new)
            .recompute_new_node_distances(i, j)
            .remove_merged_nodes(i, j);
    }
    tree.complete = true;
    tree.create_postorder();
    tree.create_preorder();
    Ok(tree)
}

fn build_nj_tree_from_matrix(nj_data: NJMat, sequences: &Vec<Record>) -> Result<Tree> {
    build_nj_tree_w_rng_from_matrix(nj_data, sequences, rng_len)
}

pub fn build_nj_tree(sequences: &Vec<Record>) -> Result<Tree> {
    let nj_data = compute_distance_matrix(sequences);
    build_nj_tree_from_matrix(nj_data, sequences)
}

fn compute_distance_matrix(sequences: &Vec<Record>) -> nj_matrices::NJMat {
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
