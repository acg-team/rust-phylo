use std::cell::RefCell;
use std::fmt::Display;

use anyhow::Ok;

use fixedbitset::FixedBitSet;
use hashbrown::HashSet;

use crate::alphabets::ParsimonySet;
use crate::likelihood::TreeSearchCost;
use crate::parsimony::scoring::{GapCost, SimpleScoring};
use crate::parsimony::ParsimonyScoring;
use crate::phylo_info::PhyloInfo;
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
    Tree,
};
use crate::Result;

#[derive(Debug, Clone)]
pub struct DolloParsimonyCost {
    info: PhyloInfo,
    tmp: RefCell<DolloParsimonyInfo>,
    scoring: Box<dyn ParsimonyScoring>,
}

impl Display for DolloParsimonyCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Dollo parsimony cost, match = 0.0, mismatch = 1.0, deletion = 1.0."
        )
    }
}

impl DolloParsimonyCost {
    pub fn new(info: PhyloInfo) -> Result<Self> {
        let tmp = RefCell::new(DolloParsimonyInfo::new(&info)?);
        Ok(DolloParsimonyCost {
            info,
            tmp,
            scoring: Box::new(SimpleScoring::new(1.0, GapCost::new(2.5, 0.5))),
        })
    }

    pub fn with_scoring(
        info: PhyloInfo,
        scoring: impl ParsimonyScoring + Clone + 'static,
    ) -> Result<Self> {
        let tmp = RefCell::new(DolloParsimonyInfo::new(&info)?);
        Ok(DolloParsimonyCost {
            info,
            tmp,
            scoring: Box::new(scoring.clone()),
        })
    }

    fn score(&self) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Leaf(_) => {
                    self.set_leaf(node_idx);
                }
                Internal(_) => {
                    self.set_internal(node_idx);
                }
            }
        }
        self.tmp.borrow().cost[usize::from(self.info.tree.root)]
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        let node = self.info.tree.node(node_idx);
        let idx = usize::from(node_idx);
        let childx_idx = usize::from(&node.children[0]);
        let childy_idx = usize::from(&node.children[1]);

        let blen = node.blen;

        if self.tmp.borrow().node_info_valid[idx] {
            return;
        }

        let mut tmp = self.tmp.borrow_mut();

        let childx_set = tmp.node_leaf_sets[childx_idx].clone();
        let childy_set = tmp.node_leaf_sets[childy_idx].clone();

        for (site_idx, (x, y)) in childx_set.iter().zip(childy_set.iter()).enumerate() {
            tmp.node_leaf_sets[idx][site_idx] = x.union(y).cloned().collect();
            if tmp.node_leaf_sets[idx][site_idx] == tmp.leaf_sets[site_idx]
                && !tmp.node_insertion[idx][site_idx]
            {
                if let Some(parent_idx) = self.info.tree.node(node_idx).parent {
                    tmp.node_insertion[usize::from(parent_idx)].set(site_idx, true);
                }
            }
        }
        let insertion = &tmp.node_insertion[idx].clone();

        let childx_sets = &tmp.node_info[childx_idx].clone();
        let childy_sets = &tmp.node_info[childy_idx].clone();
        let mut cost = tmp.cost[childx_idx] + tmp.cost[childy_idx];

        let gap = self.info.msa.alphabet().gap_set();
        let node_info: Vec<ParsimonySet> = childx_sets
            .iter()
            .zip(childy_sets)
            .enumerate()
            .map(|(i, (x_set, y_set))| {
                let intersection = x_set & y_set;
                if intersection.is_empty() {
                    if !insertion[i] {
                        let union = x_set | y_set;
                        if (&union & gap).is_empty() {
                            cost += self.scoring.min_match(blen, x_set, y_set);
                        } else {
                            cost += self.scoring.gap_open(blen);
                        }
                        &(x_set | y_set) - gap
                    } else {
                        gap.clone()
                    }
                } else {
                    cost += self.scoring.min_match(blen, &intersection, &intersection);
                    intersection
                }
            })
            .collect();

        tmp.cost[idx] = cost;
        if let Some(parent_idx) = node.parent {
            tmp.node_info_valid[usize::from(parent_idx)] = false;
        }
        tmp.node_info[idx] = node_info;
        tmp.node_info_valid[idx] = true;
        drop(tmp);
    }

    fn set_leaf(&self, node_idx: &NodeIdx) {
        let node = self.info.tree.node(node_idx);
        let idx = usize::from(node_idx);
        let mut tmp = self.tmp.borrow_mut();

        if !tmp.node_info_valid[idx] {
            if let Some(parent_idx) = node.parent {
                tmp.node_info_valid[usize::from(parent_idx)] = false;
            }
        }
        tmp.node_info_valid[idx] = true;
        drop(tmp);
    }
}

impl TreeSearchCost for DolloParsimonyCost {
    fn cost(&self) -> f64 {
        -self.score()
    }

    fn update_tree(&mut self, tree: Tree, dirty_nodes: &[NodeIdx]) {
        self.info.tree = tree;
        if dirty_nodes.is_empty() {
            self.tmp.borrow_mut().node_info_valid.fill(false);
            return;
        }
        for node_idx in dirty_nodes {
            self.tmp.borrow_mut().node_info_valid[usize::from(node_idx)] = false;
        }
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DolloParsimonyInfo {
    node_info: Vec<Vec<ParsimonySet>>,
    node_info_valid: Vec<bool>,
    node_insertion: Vec<FixedBitSet>,
    leaf_sets: Vec<HashSet<usize>>,
    node_leaf_sets: Vec<Vec<HashSet<usize>>>,
    cost: Vec<f64>,
}

impl DolloParsimonyInfo {
    pub fn new(info: &PhyloInfo) -> Result<Self> {
        let node_count = info.tree.len();
        let leaf_count = info.tree.leaves().len();
        let msa_length = info.msa.len();

        let mut node_info = vec![vec![ParsimonySet::empty(); msa_length]; node_count];
        let mut node_insertion = vec![FixedBitSet::with_capacity(msa_length); node_count];

        let mut leaf_sets = vec![HashSet::with_capacity(leaf_count); msa_length];
        let mut node_leaf_sets =
            vec![vec![HashSet::with_capacity(leaf_count); msa_length]; node_count];

        for node in info.tree.leaves() {
            let seq = info.msa.seqs.record_by_id(&node.id).seq().to_vec();
            let leaf_map = info.msa.leaf_map(&node.idx);
            let node_idx = usize::from(node.idx);

            node_insertion[node_idx].set_range(.., false);

            for (idx, &site) in leaf_map.iter().enumerate() {
                if let Some(pos) = site {
                    node_info[node_idx][idx] = info.msa.alphabet().parsimony_set(&seq[pos]).clone();
                    leaf_sets[idx].insert(node_idx);
                    node_leaf_sets[node_idx][idx].insert(node_idx);
                } else {
                    node_info[node_idx][idx] = info.msa.alphabet().gap_set().clone();
                }
            }
        }

        Ok(DolloParsimonyInfo {
            node_info,
            node_info_valid: vec![false; node_count],
            node_insertion,
            leaf_sets,
            node_leaf_sets,
            cost: vec![0.0; node_count],
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use super::*;

    use crate::alignment::{Alignment, Sequences};
    use crate::phylo_info::PhyloInfo;
    use crate::{record_wo_desc as record, tree};

    #[test]
    fn repeat_dollo_parsimony_score() {
        let seqs = Sequences::new(vec![
            record!("A", b"G-GA"),
            record!("B", b"G-GG"),
            record!("C", b"AGCA"),
            record!("D", b"AGCG"),
        ]);

        let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

        let info = PhyloInfo {
            msa: Alignment::from_aligned(seqs.clone(), &tree).unwrap(),
            tree,
        };
        let cost = DolloParsimonyCost::new(info).unwrap();
        assert_eq!(cost.score(), 4.0);
        assert_eq!(cost.score(), 4.0);

        cost.tmp.borrow_mut().node_info_valid.fill(false);
        assert_eq!(cost.score(), 4.0);
    }
}
