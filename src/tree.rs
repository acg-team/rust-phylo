pub(crate) mod njmat;

use crate::Result;
use anyhow::bail;
use bio::alignment::distance::levenshtein;
use bio::io::fasta;
use log::info;
use nalgebra::max;
use nalgebra::DMatrix;
use njmat::{Mat, NJMat};
use pest::{error::Error as PestError, iterators::Pair, Parser};
use pest_derive::Parser;
use std::error::Error;
use std::fmt;
use std::result::Result as stdResult;

#[derive(Parser)]
#[grammar = "newick.pest"]
pub struct NewickParser;

#[derive(Debug)]
pub(crate) struct ParsingError(PestError<Rule>);

impl fmt::Display for ParsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Malformed newick string")?;
        write!(f, "{}", self.0)
    }
}
impl Error for ParsingError {}

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Eq, Ord, Hash)]
pub(crate) enum NodeIdx {
    Internal(usize),
    Leaf(usize),
}

use rand::random;
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

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct Node {
    pub(crate) idx: NodeIdx,
    pub(crate) parent: Option<NodeIdx>,
    pub(crate) children: Vec<NodeIdx>,
    pub(crate) blen: f64,
    pub(crate) id: String,
}

impl Node {
    fn new_empty_leaf(node_idx: usize) -> Self {
        Self::new_leaf(node_idx, None, 0.0, "".to_string())
    }

    fn new_leaf(idx: usize, parent: Option<NodeIdx>, blen: f64, id: String) -> Self {
        Self {
            idx: Leaf(idx),
            parent,
            children: Vec::new(),
            blen,
            id: id,
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
            children: children.clone(),
            blen,
            id: id,
        }
    }

    fn new_empty_internal(node_idx: usize) -> Self {
        Self::new_internal(node_idx, None, Vec::new(), 0.0, "".to_string())
    }

    pub(crate) fn add_parent(&mut self, parent_idx: NodeIdx, blen: f64) {
        assert!(matches!(parent_idx, Int(_)));
        self.parent = Some(parent_idx);
        self.blen = blen;
    }
}

#[derive(Debug)]
pub(crate) struct Tree {
    pub(crate) root: NodeIdx,
    pub(crate) leaves: Vec<Node>,
    pub(crate) internals: Vec<Node>,
    pub(crate) postorder: Vec<NodeIdx>,
    pub(crate) preorder: Vec<NodeIdx>,
}

pub(crate) fn from_newick_string(newick_string: &str) -> Result<Vec<Tree>> {
    info!("Parsing newick trees.");
    let mut trees = Vec::new();
    let newick_tree_res = NewickParser::parse(Rule::newick, newick_string);
    if newick_tree_res.is_err() {
        bail!(ParsingError(newick_tree_res.err().unwrap()));
    }
    let newick_tree_rule = newick_tree_res.unwrap().next().unwrap();
    match newick_tree_rule.as_rule() {
        Rule::newick => {
            for tree_rule in newick_tree_rule.into_inner() {
                let tmp = tree_rule.into_inner().next();
                if let Some(rule) = tmp {
                    let mut tree = Tree::new_empty();
                    let res = tree.from_tree_rule(rule);
                    if res.is_err() {
                        bail!(ParsingError(res.err().unwrap()));
                    }
                    trees.push(tree);
                }
            }
        }
        _ => unimplemented!(),
    }
    info!("Finished parsing newick trees successfully.");
    Ok(trees)
}

impl Tree {
    pub(crate) fn new_empty() -> Self {
        Self {
            root: Int(0),
            leaves: Vec::new(),
            internals: Vec::new(),
            postorder: Vec::new(),
            preorder: Vec::new(),
        }
    }

    fn from_tree_rule(&mut self, tree_rule: Pair<Rule>) -> stdResult<(), PestError<Rule>> {
        let mut leaf_idx = 0;
        let mut internal_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        match tree_rule.as_rule() {
            Rule::internal => {
                self.from_internal_rule(
                    &mut leaf_idx,
                    &mut internal_idx,
                    &mut parent_stack,
                    tree_rule,
                )?;
            }
            Rule::leaf => {
                self.from_leaf_rule(&mut leaf_idx, tree_rule)?;
                self.root = Leaf(0);
            }
            _ => unreachable!(),
        }
        self.create_postorder();
        self.create_preorder();
        Ok(())
    }

    fn from_internal_rule(
        &mut self,
        leaf_idx: &mut usize,
        node_idx: &mut usize,
        stack: &mut Vec<usize>,
        internal_rule: Pair<Rule>,
    ) -> stdResult<(), PestError<Rule>> {
        let mut id = String::from("");
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();
        stack.push(*node_idx);
        self.internals.push(Node::new_empty_internal(*node_idx));
        *node_idx += 1;
        for rule in internal_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Tree::from_label_rule(rule),
                Rule::branch_length => blen = Tree::from_branch_length_rule(rule),
                Rule::internal => {
                    children.push(Int(*node_idx));
                    self.from_internal_rule(leaf_idx, node_idx, stack, rule)?;
                }
                Rule::leaf => {
                    children.push(Leaf(*leaf_idx));
                    self.from_leaf_rule(leaf_idx, rule)?;
                    *leaf_idx += 1;
                }
                _ => unreachable!(),
            }
        }
        let cur_node_idx = stack.pop().unwrap_or_default();
        self.internals[cur_node_idx].id = id;
        self.internals[cur_node_idx].blen = blen;
        self.internals[cur_node_idx].children = children.clone();
        for child_idx in &children {
            match child_idx {
                Int(idx) => self.internals[*idx].parent = Some(Int(cur_node_idx)),
                Leaf(idx) => self.leaves[*idx].parent = Some(Int(cur_node_idx)),
            }
        }
        Ok(())
    }

    fn from_leaf_rule(
        &mut self,
        node_idx: &usize,
        inner_rule: Pair<Rule>,
    ) -> stdResult<(), PestError<Rule>> {
        let mut id = String::from("");
        let mut blen = 0.0;
        for rule in inner_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Tree::from_label_rule(rule),
                Rule::branch_length => blen = Tree::from_branch_length_rule(rule),
                _ => unreachable!(),
            }
        }
        self.leaves.push(Node::new_leaf(*node_idx, None, blen, id));
        Ok(())
    }

    fn from_branch_length_rule(rule: Pair<Rule>) -> f64 {
        rule.into_inner()
            .next()
            .unwrap()
            .as_str()
            .parse::<f64>()
            .unwrap_or_default()
    }

    fn from_label_rule(rule: Pair<Rule>) -> String {
        rule.as_str().to_string()
    }
}

impl Tree {
    pub(crate) fn new(n: usize, root: usize) -> Self {
        Self {
            root: Int(root),
            postorder: Vec::new(),
            preorder: Vec::new(),
            leaves: (0..n).map(Node::new_empty_leaf).collect(),
            internals: Vec::with_capacity(n - 1),
        }
    }

    pub(crate) fn add_parent(
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

    fn add_parent_to_child(&mut self, idx: &NodeIdx, parent_idx: usize, blen: f64) {
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

    pub(crate) fn get_leaf_ids(&self) -> Vec<String> {
        self.leaves.iter().map(|node| node.id.clone()).collect()
    }
}

#[allow(dead_code)]
fn argmin_wo_diagonal(q: Mat, rng: fn(usize) -> usize) -> (usize, usize) {
    assert!(!q.is_empty(), "The input matrix must not be empty.");
    assert!(
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

#[allow(dead_code)]
fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}

#[allow(dead_code)]
pub(crate) fn build_nj_tree_w_rng_from_matrix(
    mut nj_data: NJMat,
    rng: fn(usize) -> usize,
) -> Result<Tree> {
    let n = nj_data.distances.ncols();
    let root_idx = n - 2;
    let mut tree = Tree::new(n, root_idx);

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
    tree.create_postorder();
    tree.create_preorder();
    Ok(tree)
}

#[allow(dead_code)]
pub(crate) fn build_nj_tree_from_matrix(nj_data: NJMat) -> Result<Tree> {
    build_nj_tree_w_rng_from_matrix(nj_data, rng_len)
}

#[allow(dead_code)]
pub(crate) fn build_nj_tree(sequences: &Vec<fasta::Record>) -> Result<Tree> {
    let nj_data = compute_distance_matrix(sequences);
    build_nj_tree_from_matrix(nj_data)
}

#[allow(dead_code)]
fn compute_distance_matrix(sequences: &Vec<fasta::Record>) -> njmat::NJMat {
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
    let nj_distances = njmat::NJMat {
        idx: (0..nseqs).map(NodeIdx::Leaf).collect(),
        distances,
    };
    nj_distances
}

#[cfg(test)]
mod tree_tests {
    use crate::tree::{
        build_nj_tree_from_matrix, build_nj_tree_w_rng_from_matrix, from_newick_string,
        njmat::NJMat, Node, NodeIdx, NodeIdx::Internal as I, NodeIdx::Leaf as L, Rule, Tree,
    };
    use approx::relative_eq;
    use nalgebra::{dmatrix};
    use pest::error::ErrorVariant;

    use super::ParsingError;

    fn setup_test_tree() -> Tree {
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
    fn subroot_preorder() {
        let tree = setup_test_tree();
        assert_eq!(tree.preorder_subroot(I(0)), [I(0), L(0), L(1)]);
        assert_eq!(tree.preorder_subroot(I(1)), [I(1), L(3), L(4)]);
        assert_eq!(tree.preorder_subroot(I(2)), [I(2), L(2), I(1), L(3), L(4)]);
        assert_eq!(
            tree.preorder_subroot(I(3)),
            [I(3), I(0), L(0), L(1), I(2), L(2), I(1), L(3), L(4)]
        );
        assert_eq!(tree.preorder_subroot(I(3)), tree.preorder);
    }

    #[test]
    fn postorder() {
        let tree = setup_test_tree();
        assert_eq!(
            tree.postorder,
            [L(0), L(1), I(0), L(2), L(3), L(4), I(1), I(2), I(3)]
        );
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

    #[test]
    fn nj_correct_web_example() {
        let nj_distances = NJMat {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 4.0, 5.0, 10.0;
                4.0, 0.0, 7.0, 12.0;
                5.0, 7.0, 0.0, 9.0;
                10.0, 12.0, 9.0, 0.0],
        };
        let nj_tree = build_nj_tree_w_rng_from_matrix(nj_distances, |_| 0).unwrap();
        let leaves = vec![
            Node::new_leaf(0, Some(I(0)), 1.0, "".to_string()),
            Node::new_leaf(1, Some(I(0)), 3.0, "".to_string()),
            Node::new_leaf(2, Some(I(1)), 2.0, "".to_string()),
            Node::new_leaf(3, Some(I(1)), 7.0, "".to_string()),
        ];
        let internals = vec![
            Node::new_internal(0, Some(I(2)), vec![L(0), L(1)], 1.0, "".to_string()),
            Node::new_internal(1, Some(I(2)), vec![L(3), L(2)], 1.0, "".to_string()),
            Node::new_internal(2, None, vec![I(0), I(1)], 0.0, "".to_string()),
        ];

        assert_eq!(nj_tree.root, I(2));
        assert_eq!(nj_tree.leaves, leaves);
        assert_eq!(nj_tree.internals, internals);
    }

    #[test]
    fn nj_correct() {
        let nj_distances = NJMat {
            idx: (0..5).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 5.0, 9.0, 9.0, 8.0;
                5.0, 0.0, 10.0, 10.0, 9.0;
                9.0, 10.0, 0.0, 8.0, 7.0;
                9.0, 10.0, 8.0, 0.0, 3.0;
                8.0, 9.0, 7.0, 3.0, 0.0],
        };
        let nj_tree = build_nj_tree_w_rng_from_matrix(nj_distances, |l| 3 % l).unwrap();
        let leaves = vec![
            Node::new_leaf(0, Some(I(0)), 2.0, "".to_string()),
            Node::new_leaf(1, Some(I(0)), 3.0, "".to_string()),
            Node::new_leaf(2, Some(I(1)), 4.0, "".to_string()),
            Node::new_leaf(3, Some(I(2)), 2.0, "".to_string()),
            Node::new_leaf(4, Some(I(2)), 1.0, "".to_string()),
        ];
        let internals = vec![
            Node::new_internal(0, Some(I(1)), vec![L(1), L(0)], 3.0, "".to_string()),
            Node::new_internal(1, Some(I(3)), vec![L(2), I(0)], 1.0, "".to_string()),
            Node::new_internal(2, Some(I(3)), vec![L(4), L(3)], 1.0, "".to_string()),
            Node::new_internal(3, None, vec![I(2), I(1)], 0.0, "".to_string()),
        ];
        assert_eq!(nj_tree.root, I(3));
        assert_eq!(nj_tree.leaves, leaves);
        assert_eq!(nj_tree.internals, internals);
    }

    #[cfg(test)]
    fn is_unique<T: std::cmp::Eq + std::hash::Hash>(vec: &Vec<T>) -> bool {
        let set: std::collections::HashSet<_> = vec.iter().collect();
        set.len() == vec.len()
    }

    #[test]
    fn protein_nj_correct() {
        // NJ based on example sequences from "./data/sequences_protein1.fasta"
        let nj_distances = NJMat {
            idx: (0..4).map(NodeIdx::Leaf).collect(),
            distances: dmatrix![
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.0, 0.0, 0.0, 0.2;
                0.2, 0.2, 0.2, 0.0],
        };
        let tree = build_nj_tree_from_matrix(nj_distances).unwrap();
        assert_eq!(tree.internals.len(), 3);
        assert_eq!(tree.postorder.len(), 7);
        assert!(is_unique(&tree.postorder));
        assert_eq!(tree.preorder.len(), 7);
        assert!(is_unique(&tree.preorder));
    }

    #[test]
    fn newick_single_correct() {
        let trees = from_newick_string(&String::from(
            "(((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0,D:1.0)G:2.0;",
        ))
        .unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, I(0));
        let leaves = vec![
            Node::new_leaf(0, Some(I(2)), 1.0, "A".to_string()),
            Node::new_leaf(1, Some(I(2)), 1.0, "B".to_string()),
            Node::new_leaf(2, Some(I(1)), 1.0, "C".to_string()),
            Node::new_leaf(3, Some(I(0)), 1.0, "D".to_string()),
        ];
        let internals = vec![
            Node::new_internal(0, None, vec![L(3), I(1)], 2.0, "G".to_string()),
            Node::new_internal(1, Some(I(0)), vec![L(2), I(2)], 1.0, "F".to_string()),
            Node::new_internal(2, Some(I(1)), vec![L(1), L(0)], 2.0, "E".to_string()),
        ];
        assert_eq!(trees[0].leaves, leaves);
        assert_eq!(trees[0].internals, internals);
        assert_eq!(trees[0].postorder.len(), 7);
        assert_eq!(trees[0].preorder.len(), 7);
    }

    #[test]
    fn newick_ladder_first_correct() {
        let trees = from_newick_string(&String::from("((A:1.0,B:1.0)E:2.0,C:1.0)F:1.0;")).unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, I(0));
        let leaves = vec![
            Node::new_leaf(0, Some(I(1)), 1.0, "A".to_string()),
            Node::new_leaf(1, Some(I(1)), 1.0, "B".to_string()),
            Node::new_leaf(2, Some(I(0)), 1.0, "C".to_string()),
        ];
        let internals = vec![
            Node::new_internal(0, None, vec![I(1), L(2)], 1.0, "F".to_string()),
            Node::new_internal(1, Some(I(0)), vec![L(1), L(0)], 2.0, "E".to_string()),
        ];
        assert_eq!(trees[0].leaves, leaves);
        assert_eq!(trees[0].internals, internals);
        assert_eq!(trees[0].postorder.len(), 5);
        assert_eq!(trees[0].preorder.len(), 5);
    }

    #[test]
    fn newick_ladder_second_correct() {
        let trees = from_newick_string(&String::from("(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;")).unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, I(0));
        let leaves = vec![
            Node::new_leaf(0, Some(I(0)), 1.0, "A".to_string()),
            Node::new_leaf(1, Some(I(1)), 1.0, "B".to_string()),
            Node::new_leaf(2, Some(I(1)), 1.0, "C".to_string()),
        ];
        let internals = vec![
            Node::new_internal(0, None, vec![I(1), L(0)], 1.0, "F".to_string()),
            Node::new_internal(1, Some(I(0)), vec![L(1), L(2)], 2.0, "E".to_string()),
        ];
        assert_eq!(trees[0].leaves, leaves);
        assert_eq!(trees[0].internals, internals);
        assert_eq!(trees[0].postorder.len(), 5);
        assert_eq!(trees[0].preorder.len(), 5);
    }

    #[test]
    fn newick_ladder_big_correct() {
        let trees = from_newick_string(&String::from(
            "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;",
        ))
        .unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, I(0));
        assert_eq!(trees[0].leaves.len(), 5);
        assert_eq!(trees[0].internals.len(), 4);
        assert_eq!(trees[0].postorder.len(), 9);
        assert_eq!(trees[0].preorder.len(), 9);
    }

    #[test]
    fn newick_complex_tree_correct() {
        // tree from file samplefraction_0.99_taxa_16_treeheight_0.8_tree1_leaves.nwk
        let trees = from_newick_string(&String::from(
            "(((15:0.0334274,4:0.0334274):0.38581,7:0.419237):0.380763,(((6:0.0973428,14:0.0973428):0.0773821,\
            (1:0.000738004,3:0.000738004):0.173987):0.548192,(((13:0.0799156,16:0.0799156):0.0667553,(5:0.123516,\
                10:0.123516):0.0231551):0.0716431,((8:0.0571164,2:0.0571164):0.0539283,(12:0.0631742,(11:0.00312848,\
                    9:0.00312848):0.0600458):0.0478705):0.107269):0.504603):0.0770827);
            ",
        ))
        .unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, I(0));
        assert_eq!(trees[0].leaves.len(), 16);
        assert_eq!(trees[0].internals.len(), 15);
    }

    #[test]
    fn newick_simple_balanced_correct() {
        let trees = from_newick_string(&String::from(
            "((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;",
        ))
        .unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, I(0));
        let leaves = vec![
            Node::new_leaf(0, Some(I(1)), 1.0, "A".to_string()),
            Node::new_leaf(1, Some(I(1)), 2.0, "B".to_string()),
            Node::new_leaf(2, Some(I(2)), 3.0, "C".to_string()),
            Node::new_leaf(3, Some(I(2)), 4.0, "D".to_string()),
        ];
        let internals = vec![
            Node::new_internal(0, None, vec![I(1), I(2)], 7.3, "G".to_string()),
            Node::new_internal(1, Some(I(0)), vec![L(1), L(0)], 5.1, "E".to_string()),
            Node::new_internal(2, Some(I(0)), vec![L(2), L(3)], 6.2, "F".to_string()),
        ];
        assert_eq!(trees[0].leaves, leaves);
        assert_eq!(trees[0].internals, internals);
        assert_eq!(trees[0].postorder.len(), 7);
        assert_eq!(trees[0].preorder.len(), 7);
    }

    #[test]
    fn newick_tiny_correct() {
        let trees = from_newick_string(&String::from("A:1.0;")).unwrap();
        assert_eq!(trees.len(), 1);
        assert_eq!(trees[0].root, L(0));
        assert_eq!(trees[0].leaves.len(), 1);
        assert_eq!(trees[0].internals.len(), 0);
    }

    #[test]
    fn newick_multiple_correct() {
        let trees = from_newick_string(&String::from(
            "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;\
            ((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;\
            (A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;",
        ))
        .unwrap();
        assert_eq!(trees.len(), 3);
        assert_eq!(trees[0].root, I(0));
        assert_eq!(trees[0].leaves.len(), 5);
        assert_eq!(trees[0].internals.len(), 4);
        assert_eq!(trees[1].root, I(0));
        assert_eq!(trees[1].leaves.len(), 4);
        assert_eq!(trees[1].internals.len(), 3);
        assert_eq!(trees[2].root, I(0));
        assert_eq!(trees[2].leaves.len(), 3);
        assert_eq!(trees[2].internals.len(), 2);
    }

    fn make_parsing_error(rules: &[Rule]) -> ErrorVariant<Rule> {
        ErrorVariant::ParsingError {
            positives: rules.to_vec(),
            negatives: vec![],
        }
    }

    fn check_parsing_error(error: anyhow::Error, expected_parsing_error: &[Rule]) {
        assert_eq!(
            error.downcast_ref::<ParsingError>().unwrap().0.variant,
            make_parsing_error(expected_parsing_error)
        );
    }

    #[test]
    fn newick_garbage() {
        let trees = from_newick_string(&String::from(";"));
        check_parsing_error(trees.unwrap_err(), &[Rule::newick]);
        let trees = from_newick_string(&String::from("()()();"));
        check_parsing_error(trees.unwrap_err(), &[Rule::internal, Rule::label]);
        let trees = from_newick_string(&String::from("((A:1.0,B:1.0);"));
        check_parsing_error(trees.unwrap_err(), &[Rule::label, Rule::branch_length]);
        let trees = from_newick_string(&String::from("((A:1.0,B:1.0));"));
        check_parsing_error(trees.unwrap_err(), &[Rule::label, Rule::branch_length]);
        let trees = from_newick_string(&String::from("(:1.0,:2.0)E:5.1;"));
        check_parsing_error(trees.unwrap_err(), &[Rule::internal, Rule::label]);
    }
}
