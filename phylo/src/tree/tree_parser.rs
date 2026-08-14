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

/// Parses Newick formatted trees from a string into a vector of `Tree` structures.
///
/// Only binary trees (rooted or unrooted) are supported, with each internal node having exactly two children.
/// Unrooted trees are automatically rooted at the trifurcation node.
/// Leaf ids are required to be unique and must start with a letter (a-z, A-Z). Duplicate leaf ids will result in an error.
/// Internal node ids are optional but should also be unique and start with a letter (a-z, A-Z) if provided.
pub fn from_newick(newick: &str) -> Result<Vec<Tree>> {
    NewickTreeParser::new().parse(newick)
}
#[derive(Debug, Clone, Copy, Default)]
pub struct NewickTreeParser {}

impl NewickTreeParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse(&self, newick: &str) -> Result<Vec<Tree>> {
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
                    if let Some(rule) = tree_rule.into_inner().next() {
                        trees.push(NewickTreeParser::parse_tree(rule)?);
                    }
                }
            }
            _ => unreachable!(),
        }
        info!("Finished parsing newick trees successfully");
        Ok(trees)
    }

    fn parse_tree(rule: Pair<Rule>) -> Result<Tree> {
        let mut tree = Self::empty_tree();
        let mut leaf_ids = HashSet::<String>::new();

        match rule.as_rule() {
            Rule::rooted => Self::parse_rooted_rule(&mut tree, &mut leaf_ids, rule)?,
            Rule::unrooted => Self::parse_unrooted_rule(&mut tree, &mut leaf_ids, rule)?,
            _ => unreachable!(),
        }

        tree.finalise();
        Ok(tree)
    }

    fn empty_tree() -> Tree {
        Tree {
            root: Int(0),
            nodes: Vec::new(),
            postorder: Vec::new(),
            preorder: Vec::new(),
            n: 0,
            length: 0.0,
            dirty: FixedBitSet::new(),
        }
    }

    fn parse_rooted_rule(
        tree: &mut Tree,
        leaf_ids: &mut HashSet<String>,
        tree_rule: Pair<Rule>,
    ) -> Result<()> {
        let node_rule = tree_rule.into_inner().next().unwrap();
        let mut node_idx = 0;
        let mut parent_stack = Vec::<usize>::new();

        match node_rule.as_rule() {
            Rule::leaf => {
                let id = Self::parse_leaf_rule(tree, &mut node_idx, node_rule)?;
                Self::insert_leaf_id(leaf_ids, id)?;
                tree.root = Leaf(0);
            }
            Rule::internal => {
                Self::parse_internal_rule(
                    tree,
                    &mut node_idx,
                    &mut parent_stack,
                    leaf_ids,
                    node_rule,
                )?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    fn parse_unrooted_rule(
        tree: &mut Tree,
        leaf_ids: &mut HashSet<String>,
        tree_rule: Pair<Rule>,
    ) -> Result<()> {
        warn!("Found unrooted tree, will root at the trifurcation");
        let mut node_idx = 0;
        let mut parent_stack = Vec::<usize>::new();
        let mut children: Vec<NodeIdx> = Vec::new();
        for node_rule in tree_rule.into_inner() {
            match node_rule.as_rule() {
                Rule::leaf => {
                    children.push(Leaf(node_idx));
                    let id = Self::parse_leaf_rule(tree, &mut node_idx, node_rule)?;
                    Self::insert_leaf_id(leaf_ids, id)?;
                }
                Rule::internal => {
                    children.push(Int(node_idx));
                    Self::parse_internal_rule(
                        tree,
                        &mut node_idx,
                        &mut parent_stack,
                        leaf_ids,
                        node_rule,
                    )?;
                }
                _ => unreachable!(),
            }
        }

        NewickTreeParser::root_at_trifurcation(tree, node_idx, children);

        Ok(())
    }

    fn insert_leaf_id(leaf_ids: &mut HashSet<String>, id: String) -> Result<()> {
        if !leaf_ids.insert(id.clone()) {
            bail!(Tree, "duplicate leaf id ({}) found in the tree", id);
        }
        Ok(())
    }

    fn parse_internal_rule(
        tree: &mut Tree,
        node_idx: &mut usize,
        stack: &mut Vec<usize>,
        leaf_ids: &mut HashSet<String>,
        internal_rule: Pair<Rule>,
    ) -> Result<()> {
        let mut id = String::from("");
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();
        stack.push(*node_idx);

        tree.nodes.push(Node::new_internal(
            *node_idx,
            None,
            vec![],
            0.0,
            "".to_string(),
        ));

        *node_idx += 1;
        for rule in internal_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Self::parse_label_rule(rule),
                Rule::branch_length => blen = Self::parse_branch_length_rule(rule),
                Rule::internal => {
                    children.push(Int(*node_idx));
                    Self::parse_internal_rule(tree, node_idx, stack, leaf_ids, rule)?;
                }
                Rule::leaf => {
                    children.push(Leaf(*node_idx));
                    let leaf_id = Self::parse_leaf_rule(tree, node_idx, rule)?;
                    Self::insert_leaf_id(leaf_ids, leaf_id)?;
                }
                _ => unreachable!(),
            }
        }
        let cur_node_idx = stack.pop().expect("newick parser stack underflow error");

        for child_idx in &children {
            tree.nodes[usize::from(child_idx)].parent = Some(Int(cur_node_idx));
        }
        tree.nodes[cur_node_idx].id = id;
        tree.nodes[cur_node_idx].blen = blen;
        tree.nodes[cur_node_idx].children = children;
        Ok(())
    }

    fn parse_leaf_rule(
        tree: &mut Tree,
        node_idx: &mut usize,
        leaf_rule: Pair<Rule>,
    ) -> Result<String> {
        let mut id = String::from("");
        let mut blen = 0.0;
        for rule in leaf_rule.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Self::parse_label_rule(rule),
                Rule::branch_length => blen = Self::parse_branch_length_rule(rule),
                _ => unreachable!(),
            }
        }
        tree.nodes
            .push(Node::new_leaf(*node_idx, None, blen, id.clone()));

        *node_idx += 1;
        Ok(id)
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

    fn root_at_trifurcation(tree: &mut Tree, mut node_idx: usize, children: Vec<NodeIdx>) {
        let new_children = children[0..2].to_vec();
        for child_idx in new_children.iter() {
            tree.nodes[usize::from(child_idx)].parent = Some(Int(node_idx));
        }

        tree.nodes.push(Node::new_internal(
            node_idx,
            None,
            new_children,
            0.0,
            "".to_string(),
        ));

        let new_children = vec![Int(node_idx), children[2]];

        node_idx += 1;
        for child_idx in new_children.iter() {
            tree.nodes[usize::from(child_idx)].parent = Some(Int(node_idx));
        }

        tree.nodes.push(Node::new_internal(
            node_idx,
            None,
            new_children,
            0.0,
            "".to_string(),
        ));
        tree.root = Int(node_idx);
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use crate::tree::tree_parser::NewickTreeParser;

    #[test]
    fn test_parser_multiple_parse_calls() {
        let parser = NewickTreeParser::new();

        let str1 = "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;\
            ((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;";
        let str2 = "(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;";

        let trees1 = parser.parse(str1).unwrap();
        assert_eq!(trees1.len(), 2);

        let trees2 = parser.parse(str2).unwrap();
        assert_eq!(trees2.len(), 1);
    }
}
