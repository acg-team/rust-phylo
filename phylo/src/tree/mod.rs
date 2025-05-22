use std::collections::HashSet;
use std::fmt::{Debug, Display};

use anyhow::bail;
use approx::relative_eq;
use inc_stats::Percentiles;

use crate::alignment::Sequences;
use crate::parsimony::Rounding;
use crate::tree::{
    NodeIdx::{Internal as Int, Leaf},
};
use crate::Result;

mod nj_matrices;
pub mod tree_parser;
pub mod tree_builder;
pub mod nj_builder;

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
    leaf_ids: Vec<String>,
    pub complete: bool,
    pub n: usize,
    pub height: f64,
    pub(crate) dirty: Vec<bool>,
}

impl Display for Tree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_newick())
    }
}

impl Tree {
    pub(crate) fn new(sequences: &Sequences) -> Result<Self> {
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
                leaf_ids: vec![sequences.record(0).id().to_string()],
                dirty: vec![false],
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
                leaf_ids: sequences.iter().map(|seq| seq.id().to_string()).collect(),
                dirty: vec![false; 2 * n - 1],
            })
        }
    }

    pub fn clean(&mut self, clean: bool) {
        self.dirty.fill(clean);
    }

    pub fn robinson_foulds(&self, other: &Tree) -> usize {
        let mut dist = 0;
        let common_leaves = self.common_leaf_set(other);
        let parts = self.filtered_partitions(&common_leaves);
        let other_parts: Vec<HashSet<String>> = other.filtered_partitions(&common_leaves);
        for part in parts.iter() {
            if !other_parts.contains(part) {
                dist += 1;
            }
        }
        dist
    }

    fn filtered_partitions(&self, common_leaves: &HashSet<String>) -> Vec<HashSet<String>> {
        self.partitions()
            .iter()
            .map(|set| set.intersection(common_leaves).cloned().collect())
            .filter(|set: &HashSet<String>| !set.is_empty())
            .collect()
    }

    fn common_leaf_set(&self, other: &Tree) -> HashSet<String> {
        HashSet::<String>::from_iter(self.leaf_ids())
            .intersection(&HashSet::from_iter(other.leaf_ids()))
            .cloned()
            .collect()
    }

    pub fn partitions(&self) -> Vec<HashSet<String>> {
        let mut partitions = Vec::new();
        let all_leaves: HashSet<String> =
            self.leaves().iter().map(|node| node.id.clone()).collect();
        // skip root because it is a trivial partition
        for node in self.preorder.iter().skip(1) {
            let partition = self
                .preorder_subroot(node)
                .iter()
                .filter(|idx| matches!(idx, Leaf(_)))
                .map(|idx| self.node_id(idx).to_string())
                .collect();
            let other: HashSet<String> = all_leaves.difference(&partition).cloned().collect();
            if partitions.contains(&partition) {
                continue;
            }
            partitions.push(partition.clone());
            partitions.push(other.clone());
        }
        partitions
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

    /// No pruning on the root branch
    pub fn find_possible_prune_locations(&self) -> impl Iterator<Item = &NodeIdx> + use<'_> {
        self.preorder().iter().filter(|&n| *n != self.root)
    }

    // TODO: Bring this out of the tree
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

    pub fn by_id(&self, id: &str) -> &Node {
        self.nodes.iter().find(|node| node.id == id).unwrap()
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
        self.leaf_ids.clone()
    }

    pub fn try_idx(&self, id: &str) -> Result<NodeIdx> {
        debug_assert!(self.complete);
        let node = self.nodes.iter().find(|node| node.id == id);
        if let Some(node) = node {
            return Ok(node.idx);
        }
        bail!("No node with id {} found in the tree", id);
    }

    #[cfg(test)]
    pub(crate) fn idx(&self, id: &str) -> NodeIdx {
        debug_assert!(self.complete);
        self.nodes.iter().find(|node| node.id == id).unwrap().idx
    }

    pub fn set_blen(&mut self, node_idx: &NodeIdx, blen: f64) {
        debug_assert!(blen >= 0.0);
        let idx = usize::from(node_idx);
        let old_blen = self.nodes[idx].blen;
        self.height += blen - old_blen;
        self.nodes[idx].blen = blen;
        self.dirty[idx] = true;
    }

    pub fn parent(&self, node_idx: &NodeIdx) -> Option<NodeIdx> {
        self.nodes[usize::from(node_idx)].parent
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
    if rounding.yes() {
        values.iter_mut().for_each(|len| {
            *len = (*len * (10.0_f64.powf(rounding.digits as f64))).round()
                / (10.0_f64.powf(rounding.digits as f64))
        });
    }
    values
}

#[cfg(not(feature = "deterministic"))]
fn rng_len(l: usize) -> usize {
    rand::random::<usize>() % l
}
#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
