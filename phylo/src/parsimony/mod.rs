use std::fmt::{self, Debug};

use log::{debug, info};
use rand::prelude::*;

use crate::alignment::{InternalMapping, PairwiseAlignment, Sequences};
use crate::tree::Tree;
use crate::tree::{NodeIdx::Internal as Int, NodeIdx::Leaf};

pub mod costs;
use costs::{BranchParsimonyCosts, ParsimonyCosts};
pub mod matrices;
pub(crate) use matrices::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapInY,
    GapInX,
}

use crate::alphabets::ParsimonySet;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SiteFlag {
    GapFixed,
    GapOpen,
    GapExt,
    NoGap,
}

#[derive(Clone, PartialEq)]
pub(crate) struct ParsimonySite {
    pub(crate) set: ParsimonySet,
    pub(super) flag: SiteFlag,
}

impl Debug for ParsimonySite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}: {:?}",
            self.set.iter().map(|&a| a as char).collect::<Vec<char>>(),
            self.flag
        )
        .unwrap();
        Ok(())
    }
}

impl ParsimonySite {
    pub(crate) fn new(set: impl IntoIterator<Item = u8>, gap_flag: SiteFlag) -> ParsimonySite {
        ParsimonySite {
            set: ParsimonySet::from_iter(set),
            flag: gap_flag,
        }
    }
    pub(crate) fn new_leaf(set: impl IntoIterator<Item = u8>) -> ParsimonySite {
        ParsimonySite::new(set, SiteFlag::NoGap)
    }

    pub(crate) fn is_fixed(&self) -> bool {
        self.flag == SiteFlag::GapFixed
    }

    #[allow(dead_code)]
    pub(crate) fn is_open(&self) -> bool {
        self.flag == SiteFlag::GapOpen
    }

    pub(crate) fn is_ext(&self) -> bool {
        self.flag == SiteFlag::GapExt
    }

    pub(crate) fn is_possible(&self) -> bool {
        self.flag == SiteFlag::GapOpen || self.flag == SiteFlag::GapExt
    }

    pub(crate) fn no_gap(&self) -> bool {
        self.flag == SiteFlag::NoGap
    }
}

fn rng_len(l: usize) -> usize {
    random::<usize>() % l
}

fn pars_align_w_rng(
    x_info: &[ParsimonySite],
    x_scoring: &dyn BranchParsimonyCosts,
    y_info: &[ParsimonySite],
    y_scoring: &dyn BranchParsimonyCosts,
    rng: fn(usize) -> usize,
) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
    let mut pars_mats = ParsimonyAlignmentMatrices::new(x_info.len() + 1, y_info.len() + 1, rng);
    debug!(
        "x_scoring: {} {} {}",
        x_scoring.avg(),
        x_scoring.gap_open(),
        x_scoring.gap_ext()
    );
    debug!(
        "y_scoring: {} {} {}",
        y_scoring.avg(),
        y_scoring.gap_open(),
        y_scoring.gap_ext()
    );
    pars_mats.fill_matrices(x_info, x_scoring, y_info, y_scoring);
    pars_mats.traceback(x_info, y_info)
}

fn pars_align(
    x_info: &[ParsimonySite],
    x_scoring: &dyn BranchParsimonyCosts,
    y_info: &[ParsimonySite],
    y_scoring: &dyn BranchParsimonyCosts,
) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
    pars_align_w_rng(x_info, x_scoring, y_info, y_scoring, rng_len)
}

pub fn pars_align_on_tree(
    scoring: &dyn ParsimonyCosts,
    tree: &Tree,
    sequences: Sequences,
) -> (InternalMapping, Vec<f64>) {
    info!("Starting the IndelMAP alignment.");
    let order = tree.postorder();

    let mut node_info = vec![Vec::<ParsimonySite>::new(); tree.len()];
    let mut alignments = InternalMapping::with_capacity(tree.internals().len());
    let mut scores = vec![0.0; tree.len()];

    for &node_idx in order {
        info!("Processing {}{}.", node_idx, tree.node(&node_idx).id);
        match node_idx {
            Int(idx) => {
                let chx_idx = tree.node(&node_idx).children[0];
                let chy_idx = tree.node(&node_idx).children[1];

                let x_info = &node_info[usize::from(chx_idx)];
                let x_branch = tree.node(&chx_idx).blen;

                let y_info = &node_info[usize::from(chy_idx)];
                let y_branch = tree.node(&chy_idx).blen;

                info!(
                    "Aligning sequences at nodes: \n1. {}{} with branch length {} \n2. {}{} with branch length {}",
                    chx_idx,
                    tree.node(&chx_idx).id,
                    x_branch,
                    chy_idx,
                    tree.node(&chy_idx).id,
                    y_branch,
                );
                let (info, alignment, score) = pars_align(
                    x_info,
                    scoring.branch_costs(x_branch),
                    y_info,
                    scoring.branch_costs(y_branch),
                );
                node_info[idx] = info;
                alignments.insert(node_idx, alignment);
                scores[idx] = score;
                info!("Alignment complete with score {}.\n", score);
            }
            Leaf(idx) => {
                let pars_sets = sequences
                    .record_by_id(tree.node_id(&node_idx))
                    .seq()
                    .iter()
                    .map(|c| sequences.alphabet().parsimony_set(c))
                    .collect::<Vec<_>>();
                node_info[idx] = pars_sets.into_iter().map(ParsimonySite::new_leaf).collect();
                info!("Processed leaf node.\n");
            }
        }
    }
    info!("Finished IndelMAP alignment.");

    (alignments, scores)
}

#[cfg(test)]
mod tests;
