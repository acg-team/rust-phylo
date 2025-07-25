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

pub trait TreeMover: Clone {
    fn tree_move_at_location<C: TreeSearchCost<Self> + Display + Send + Clone + Display>(
        &self,
        base_cost: f64,
        cost: &C,
        node_idx: &NodeIdx,
    ) -> Result<Option<MoveCostInfo>>;

    // or should we also use the cost here such that it might be more likely that this method and
    // the method above actually use the same tree
    fn move_locations<'a>(&self, tree: &'a Tree) -> impl Iterator<Item = &'a NodeIdx>;
}
