use super::Node;
use super::NodeIdx::{self, Internal as Int, Leaf};
use super::Tree;
use crate::Result;
use anyhow::bail;
use log::info;
use pest::{error::Error as PestError, iterators::Pair, Parser};
use pest_derive::Parser;
use std::fmt;
use std::result::Result as stdResult;

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

pub(crate) fn from_newick_string(newick_string: &str) -> Result<Vec<Tree>> {
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
                    let res = tree.parse_tree_rule(rule);
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
            leaves: Vec::new(),
            internals: Vec::new(),
            postorder: Vec::new(),
            preorder: Vec::new(),
        }
    }

    fn parse_tree_rule(&mut self, tree_rule: Pair<Rule>) -> stdResult<(), Box<PestError<Rule>>> {
        let mut leaf_idx = 0;
        let mut internal_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        match tree_rule.as_rule() {
            Rule::internal => {
                self.parse_internal_rule(
                    &mut leaf_idx,
                    &mut internal_idx,
                    &mut parent_stack,
                    tree_rule,
                )?;
            }
            Rule::leaf => {
                self.parse_leaf_rule(&leaf_idx, tree_rule)?;
                self.root = Leaf(0);
            }
            _ => unreachable!(),
        }
        self.create_postorder();
        self.create_preorder();
        Ok(())
    }

    fn parse_internal_rule(
        &mut self,
        leaf_idx: &mut usize,
        node_idx: &mut usize,
        stack: &mut Vec<usize>,
        internal_rule: Pair<Rule>,
    ) -> stdResult<(), Box<PestError<Rule>>> {
        let mut id = String::from("");
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();
        stack.push(*node_idx);
        self.internals.push(Node::new_empty_internal(*node_idx));
        *node_idx += 1;
        for rule in internal_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Tree::parse_label_rule(rule),
                Rule::branch_length => blen = Tree::parse_branch_length_rule(rule),
                Rule::internal => {
                    children.push(Int(*node_idx));
                    self.parse_internal_rule(leaf_idx, node_idx, stack, rule)?;
                }
                Rule::leaf => {
                    children.push(Leaf(*leaf_idx));
                    self.parse_leaf_rule(leaf_idx, rule)?;
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

    fn parse_leaf_rule(
        &mut self,
        node_idx: &usize,
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
        self.leaves.push(Node::new_leaf(*node_idx, None, blen, id));
        Ok(())
    }

    fn parse_branch_length_rule(rule: Pair<Rule>) -> f64 {
        rule.into_inner()
            .next()
            .unwrap()
            .as_str()
            .parse::<f64>()
            .unwrap_or_default()
    }

    fn parse_label_rule(rule: Pair<Rule>) -> String {
        rule.as_str().to_string()
    }
}
