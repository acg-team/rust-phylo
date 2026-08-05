use fixedbitset::FixedBitSet;
use hashbrown::HashSet;
use log::{info, warn};
use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

use crate::bail;
use crate::tree::{
    Node,
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

#[derive(Parser)]
#[grammar = "./tree/newick.pest"]
pub struct NewickParser;

/// Only binary trees (rooted or unrooted) are supported.
pub fn from_newick(newick: &str) -> Result<Vec<Tree>> {
    NewickTreeParser::new().parse(newick)
}

pub struct NewickTreeParser {
    leaf_ids: HashSet<String>,
}

impl Default for NewickTreeParser {
    fn default() -> Self {
        Self {
            leaf_ids: HashSet::new(),
        }
    }
}

impl NewickTreeParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse(mut self, newick: &str) -> Result<Vec<Tree>> {
        info!("Parsing newick trees");
        let mut trees = Vec::new();
        let newick_tree_res = NewickParser::parse(Rule::newick, newick);
        if let Err(e) = newick_tree_res {
            bail!(TreeParsing, "malformed newick string", Box::new(e));
        }

        let newick_tree_rule = newick_tree_res.unwrap().next().unwrap();
        match newick_tree_rule.as_rule() {
            Rule::newick => {
                for tree_rule in newick_tree_rule.into_inner() {
                    self.leaf_ids.clear();
                    let tmp = tree_rule.into_inner().next();
                    if let Some(rule) = tmp {
                        let mut tree = self.new_tree();
                        match rule.as_rule() {
                            Rule::rooted => self.parse_rooted_rule(&mut tree, rule)?,
                            Rule::unrooted => self.parse_unrooted_rule(&mut tree, rule)?,
                            _ => unimplemented!(),
                        };
                        trees.push(tree);
                    }
                }
            }
            _ => unimplemented!(),
        }
        info!("Finished parsing newick trees successfully");
        Ok(trees)
    }

    fn new_tree(&self) -> Tree {
        Tree {
            root: Int(0),
            nodes: Vec::new(),
            postorder: Vec::new(),
            preorder: Vec::new(),
            complete: false,
            n: 0,
            length: 0.0,
            dirty: FixedBitSet::new(),
        }
    }

    fn parse_rooted_rule(&mut self, tree: &mut Tree, node_rule: Pair<Rule>) -> Result<()> {
        let tree_rule = node_rule.into_inner().next().unwrap();
        let mut node_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        match tree_rule.as_rule() {
            Rule::leaf => {
                self.parse_leaf_rule(tree, &mut node_idx, tree_rule)?;
                tree.root = Leaf(0);
            }
            Rule::internal => {
                self.parse_internal_rule(tree, &mut node_idx, &mut parent_stack, tree_rule)?;
            }
            _ => unreachable!(),
        }

        self.complete(tree);
        Ok(())
    }

    fn complete(&mut self, tree: &mut Tree) {
        tree.n = tree.nodes.len().div_ceil(2);
        debug_assert_eq!(tree.nodes.len(), tree.n * 2 - 1);
        tree.complete = true;
        tree.compute_postorder();
        tree.compute_preorder();
        tree.length = tree.nodes.iter().map(|n| n.blen).sum();
        tree.dirty = FixedBitSet::with_capacity(tree.n * 2 - 1);
    }

    fn parse_unrooted_rule(&mut self, tree: &mut Tree, tree_rule: Pair<Rule>) -> Result<()> {
        warn!("Found unrooted tree, will root at the trifurcation");
        let mut node_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        let mut children: Vec<NodeIdx> = Vec::new();
        for node_rule in tree_rule.into_inner() {
            match node_rule.as_rule() {
                Rule::leaf => {
                    children.push(Leaf(node_idx));
                    self.parse_leaf_rule(tree, &mut node_idx, node_rule)?;
                }
                Rule::internal => {
                    children.push(Int(node_idx));
                    self.parse_internal_rule(tree, &mut node_idx, &mut parent_stack, node_rule)?;
                }
                _ => unreachable!(),
            }
        }

        tree.nodes.push(Node::new_empty_internal(node_idx));
        let new_children = children[0..2].to_vec();
        for child_idx in new_children.iter() {
            tree.add_parent_to_child_no_blen(child_idx, &Int(node_idx));
        }
        tree.nodes[node_idx].children = new_children;
        node_idx += 1;

        tree.nodes.push(Node::new_empty_internal(node_idx));
        let new_children = vec![Int(node_idx - 1), children[2]];
        for child_idx in new_children.iter() {
            tree.add_parent_to_child_no_blen(child_idx, &Int(node_idx));
        }
        tree.nodes[node_idx].children = new_children;
        tree.root = Int(node_idx);

        self.complete(tree);
        Ok(())
    }

    fn parse_internal_rule(
        &mut self,
        tree: &mut Tree,
        node_idx: &mut usize,
        stack: &mut Vec<usize>,
        internal_rule: Pair<Rule>,
    ) -> Result<()> {
        let mut id = String::from("");
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();
        stack.push(*node_idx);
        tree.nodes.push(Node::new_empty_internal(*node_idx));
        *node_idx += 1;
        for rule in internal_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Self::parse_label_rule(rule),
                Rule::branch_length => blen = Self::parse_branch_length_rule(rule),
                Rule::internal => {
                    children.push(Int(*node_idx));
                    self.parse_internal_rule(tree, node_idx, stack, rule)?;
                }
                Rule::leaf => {
                    children.push(Leaf(*node_idx));
                    self.parse_leaf_rule(tree, node_idx, rule)?;
                }
                _ => unreachable!(),
            }
        }
        let cur_node_idx = stack.pop().unwrap_or_default();
        tree.nodes[cur_node_idx].id = id;
        tree.nodes[cur_node_idx].blen = blen;
        tree.nodes[cur_node_idx].children.clone_from(&children);
        for child_idx in &children {
            match child_idx {
                Int(idx) => tree.nodes[*idx].parent = Some(Int(cur_node_idx)),
                Leaf(idx) => tree.nodes[*idx].parent = Some(Int(cur_node_idx)),
            }
        }
        Ok(())
    }

    fn parse_leaf_rule(
        &mut self,
        tree: &mut Tree,
        node_idx: &mut usize,
        inner_rule: Pair<Rule>,
    ) -> Result<()> {
        let mut id = String::from("");
        let mut blen = 0.0;
        for rule in inner_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Self::parse_label_rule(rule),
                Rule::branch_length => blen = Self::parse_branch_length_rule(rule),
                _ => unreachable!(),
            }
        }
        tree.nodes
            .push(Node::new_leaf(*node_idx, None, blen, id.clone()));
        if !self.leaf_ids.insert(id.clone()) {
            bail!(Tree, "duplicate leaf id ({}) found in the tree", id);
        }
        *node_idx += 1;
        Ok(())
    }

    fn parse_branch_length_rule(rule: Pair<Rule>) -> f64 {
        rule.into_inner()
            .next()
            .unwrap()
            .as_str()
            .trim()
            .parse::<f64>()
            .unwrap_or_default()
    }

    fn parse_label_rule(rule: Pair<Rule>) -> String {
        rule.as_str().to_string()
    }
}
