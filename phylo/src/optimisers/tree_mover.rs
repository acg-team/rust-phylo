use std::fmt::Display;

use crate::likelihood::TreeSearchCost;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

pub struct MoveCostInfo {
    cost: f64,
    tree: Tree,
    dirty_nodes: Vec<NodeIdx>,
}

impl MoveCostInfo {
    pub(crate) fn new(cost: f64, tree: Tree, dirty_nodes: Vec<NodeIdx>) -> Self {
        MoveCostInfo {
            cost,
            dirty_nodes,
            tree,
        }
    }
    pub fn dirty_nodes(&self) -> &Vec<NodeIdx> {
        &self.dirty_nodes
    }
    pub fn cost(&self) -> f64 {
        self.cost
    }
    pub fn tree(&self) -> &Tree {
        &self.tree
    }
    pub fn into_tree(self) -> Tree {
        self.tree
    }
}

pub trait TreeMover {
    fn tree_move_at_node<C: TreeSearchCost + Display + Send + Clone + Display>(
        &self,
        base_cost: f64,
        cost: &C,
        node: &NodeIdx,
    ) -> Result<Option<MoveCostInfo>>;
}
