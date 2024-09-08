use std::fmt;
use std::result::Result as stdResult;

use anyhow::bail;
use log::{info, warn};
use pest::{error::Error as PestError, iterators::Pair, Parser};
use pest_derive::Parser;

use crate::tree::{
    Node,
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

#[derive(Parser)]
#[grammar = "./tree/newick.pest"]
pub struct NewickParser;

#[derive(Debug)]
pub(crate) struct ParsingError(pub(crate) Box<PestError<Rule>>);

impl fmt::Display for ParsingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Malformed newick string")?;
        write!(f, "{}", self.0)
    }
}

pub fn from_newick_string(newick_string: &str) -> Result<Vec<Tree>> {
    info!("Parsing newick trees.");
    let mut trees = Vec::new();
    let newick_tree_res = NewickParser::parse(Rule::newick, newick_string);
    if newick_tree_res.is_err() {
        bail!(ParsingError(Box::new(newick_tree_res.err().unwrap())));
    }
    let newick_tree_rule = newick_tree_res.unwrap().next().unwrap();
    match newick_tree_rule.as_rule() {
        Rule::newick => {
            for tree_rule in newick_tree_rule.into_inner() {
                let tmp = tree_rule.into_inner().next();
                if let Some(rule) = tmp {
                    let mut tree = Tree::new_empty();
                    let res = match rule.as_rule() {
                        Rule::rooted => tree.parse_rooted_rule(rule),
                        Rule::unrooted => tree.parse_unrooted_rule(rule),
                        _ => unimplemented!(),
                    };
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
    fn new_empty() -> Self {
        Self {
            root: Int(0),
            nodes: Vec::new(),
            postorder: Vec::new(),
            preorder: Vec::new(),
            complete: false,
            n: 0,
            height: 0.0,
            leaf_ids: Vec::new(),
            dirty: Vec::new(),
        }
    }

    fn parse_rooted_rule(&mut self, node_rule: Pair<Rule>) -> stdResult<(), Box<PestError<Rule>>> {
        let tree_rule = node_rule.into_inner().next().unwrap();
        let mut node_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        match tree_rule.as_rule() {
            Rule::leaf => {
                self.parse_leaf_rule(&mut node_idx, tree_rule)?;
                self.root = Leaf(0);
            }
            Rule::internal => {
                self.parse_internal_rule(&mut node_idx, &mut parent_stack, tree_rule)?;
            }
            _ => unreachable!(),
        }

        self.complete();
        Ok(())
    }

    fn complete(&mut self) {
        self.n = (self.nodes.len() + 1) / 2;
        debug_assert_eq!(self.nodes.len(), self.n * 2 - 1);
        self.complete = true;
        self.compute_postorder();
        self.compute_preorder();
        self.height = self.nodes.iter().map(|n| n.blen).sum();
        self.dirty = vec![false; self.n * 2 - 1];
    }

    fn parse_unrooted_rule(
        &mut self,
        tree_rule: Pair<Rule>,
    ) -> stdResult<(), Box<PestError<Rule>>> {
        warn!("Found unrooted tree, will root at the trifurcation.");
        let mut node_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        let mut children: Vec<NodeIdx> = Vec::new();
        for node_rule in tree_rule.into_inner() {
            match node_rule.as_rule() {
                Rule::leaf => {
                    children.push(Leaf(node_idx));
                    self.parse_leaf_rule(&mut node_idx, node_rule)?;
                }
                Rule::internal => {
                    children.push(Int(node_idx));
                    self.parse_internal_rule(&mut node_idx, &mut parent_stack, node_rule)?;
                }
                _ => unreachable!(),
            }
        }

        self.nodes.push(Node::new_empty_internal(node_idx));
        let new_children = children[0..2].to_vec();
        for child_idx in new_children.iter() {
            self.add_parent_to_child_no_blen(child_idx, &Int(node_idx));
        }
        self.nodes[node_idx].children = new_children;
        node_idx += 1;

        self.nodes.push(Node::new_empty_internal(node_idx));
        let new_children = vec![Int(node_idx - 1), children[2]];
        for child_idx in new_children.iter() {
            self.add_parent_to_child_no_blen(child_idx, &Int(node_idx));
        }
        self.nodes[node_idx].children = new_children;
        self.root = Int(node_idx);

        self.complete();
        Ok(())
    }

    fn parse_internal_rule(
        &mut self,
        node_idx: &mut usize,
        stack: &mut Vec<usize>,
        internal_rule: Pair<Rule>,
    ) -> stdResult<(), Box<PestError<Rule>>> {
        let mut id = String::from("");
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();
        stack.push(*node_idx);
        self.nodes.push(Node::new_empty_internal(*node_idx));
        *node_idx += 1;
        for rule in internal_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Tree::parse_label_rule(rule),
                Rule::branch_length => blen = Tree::parse_branch_length_rule(rule),
                Rule::internal => {
                    children.push(Int(*node_idx));
                    self.parse_internal_rule(node_idx, stack, rule)?;
                }
                Rule::leaf => {
                    children.push(Leaf(*node_idx));
                    self.parse_leaf_rule(node_idx, rule)?;
                }
                _ => unreachable!(),
            }
        }
        let cur_node_idx = stack.pop().unwrap_or_default();
        self.nodes[cur_node_idx].id = id;
        self.nodes[cur_node_idx].blen = blen;
        self.nodes[cur_node_idx].children.clone_from(&children);
        for child_idx in &children {
            match child_idx {
                Int(idx) => self.nodes[*idx].parent = Some(Int(cur_node_idx)),
                Leaf(idx) => self.nodes[*idx].parent = Some(Int(cur_node_idx)),
            }
        }
        Ok(())
    }

    fn parse_leaf_rule(
        &mut self,
        node_idx: &mut usize,
        inner_rule: Pair<Rule>,
    ) -> stdResult<(), Box<PestError<Rule>>> {
        let mut id = String::from("");
        let mut blen = 0.0;
        for rule in inner_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Tree::parse_label_rule(rule),
                Rule::branch_length => blen = Tree::parse_branch_length_rule(rule),
                _ => unreachable!(),
            }
        }
        self.nodes
            .push(Node::new_leaf(*node_idx, None, blen, id.clone()));
        self.leaf_ids.push(id);
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
