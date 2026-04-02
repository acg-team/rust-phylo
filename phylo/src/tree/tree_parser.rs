use std::result::Result as stdResult;

use fixedbitset::FixedBitSet;
use log::{info, warn};
use pest::{error::Error as PestError, iterators::Pair, Parser};
use pest_derive::Parser;

use crate::bail;
use crate::tree::{
    generate_internal_node_id, Node,
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

#[derive(Parser)]
#[grammar = "./tree/newick.pest"]
pub struct NewickParser;

/// Only binary trees (rooted or unrooted) are supported.
pub fn from_newick(newick: &str) -> Result<Vec<Tree>> {
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
                let next_rule = tree_rule.into_inner().next();
                if let Some(rule) = next_rule {
                    let mut tree = Tree::new_empty();
                    let res = match rule.as_rule() {
                        Rule::rooted => tree.parse_rooted_rule(rule),
                        Rule::unrooted => tree.parse_unrooted_rule(rule),
                        _ => unimplemented!(),
                    };
                    if let Err(e) = res {
                        bail!(TreeParsing, "malformed newick string", e);
                    }

                    tree.node_ids_are_unique()?;
                    trees.push(tree);
                }
            }
        }
        _ => unimplemented!(),
    }
    info!("Finished parsing newick trees successfully");
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
            length: 0.0,
            leaf_ids: Vec::new(),
            dirty: FixedBitSet::new(),
        }
    }

    fn parse_rooted_rule(&mut self, node_rule: Pair<Rule>) -> stdResult<(), Box<PestError<Rule>>> {
        let tree_rule = node_rule.into_inner().next().unwrap();
        let mut node_idx = 0;
        match tree_rule.as_rule() {
            Rule::leaf => {
                self.parse_leaf_rule(&node_idx, tree_rule)?;
                self.root = Leaf(0);
            }
            Rule::internal => {
                let root_idx = self.parse_internal_rule(&mut node_idx, tree_rule)?;
                self.root = Int(root_idx);
            }
            _ => unreachable!(),
        }

        self.complete();
        Ok(())
    }

    fn complete(&mut self) {
        self.n = self.nodes.len().div_ceil(2);
        debug_assert_eq!(self.nodes.len(), self.n * 2 - 1);
        self.complete = true;
        self.compute_postorder();
        self.compute_preorder();
        self.length = self.nodes.iter().map(|n| n.blen).sum();
        self.dirty = FixedBitSet::with_capacity(self.n * 2 - 1);
    }

    fn parse_unrooted_rule(
        &mut self,
        tree_rule: Pair<Rule>,
    ) -> stdResult<(), Box<PestError<Rule>>> {
        warn!("Found unrooted tree, will root at the trifurcation");
        let mut node_idx = 0;
        let mut children: Vec<NodeIdx> = Vec::new();
        for node_rule in tree_rule.into_inner() {
            match node_rule.as_rule() {
                Rule::leaf => {
                    let child = self.parse_leaf_rule(&node_idx, node_rule)?;
                    children.push(Leaf(child));
                    node_idx += 1;
                }
                Rule::internal => {
                    let child = self.parse_internal_rule(&mut node_idx, node_rule)?;
                    children.push(Int(child));
                    node_idx += 1;
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
        self.nodes[node_idx].id = generate_internal_node_id(&node_idx);
        node_idx += 1;

        self.nodes.push(Node::new_empty_internal(node_idx));
        let new_children = vec![Int(node_idx - 1), children[2]];
        for child_idx in new_children.iter() {
            self.add_parent_to_child_no_blen(child_idx, &Int(node_idx));
        }
        self.nodes[node_idx].children = new_children;
        self.nodes[node_idx].id = generate_internal_node_id(&node_idx);

        self.root = Int(node_idx);

        self.complete();
        Ok(())
    }

    fn parse_internal_rule(
        &mut self,
        node_idx: &mut usize,
        internal_rule: Pair<Rule>,
    ) -> stdResult<usize, Box<PestError<Rule>>> {
        let mut parsed_id = None;
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();

        for rule in internal_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => parsed_id = Some(Tree::parse_label_rule(rule)),
                Rule::support => {} // branch support value, ignored
                Rule::branch_length => blen = Tree::parse_branch_length_rule(rule),
                Rule::internal => {
                    let child = self.parse_internal_rule(node_idx, rule)?;
                    children.push(Int(child));
                    *node_idx += 1;
                }
                Rule::leaf => {
                    let child = self.parse_leaf_rule(node_idx, rule)?;
                    children.push(Leaf(child));
                    *node_idx += 1;
                }
                _ => unreachable!(),
            }
        }

        for child_idx in &children {
            self.add_parent_to_child_no_blen(child_idx, &Int(*node_idx));
        }

        let id = if let Some(parsed_id) = parsed_id {
            parsed_id
        } else {
            generate_internal_node_id(node_idx)
        };
        let node = Node::new_internal(*node_idx, None, children, blen, id);

        self.nodes.push(node);

        Ok(*node_idx)
    }

    fn parse_leaf_rule(
        &mut self,
        node_idx: &usize,
        inner_rule: Pair<Rule>,
    ) -> stdResult<usize, Box<PestError<Rule>>> {
        let mut parsed_id = None;
        let mut blen = 0.0;

        for rule in inner_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => parsed_id = Some(Tree::parse_label_rule(rule)),
                Rule::branch_length => blen = Tree::parse_branch_length_rule(rule),
                _ => unreachable!(),
            }
        }

        let id = if let Some(parsed_id) = parsed_id {
            parsed_id
        } else {
            unreachable!("leaf node missing id")
        };

        self.nodes
            .push(Node::new_leaf(*node_idx, None, blen, id.clone()));
        self.leaf_ids.push(id);

        Ok(*node_idx)
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
