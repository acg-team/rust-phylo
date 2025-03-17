use std::fmt::{self, Debug, Display};

use log::info;
use nalgebra::DMatrix;
use rand::prelude::*;

use crate::alignment::{InternalMapping, PairwiseAlignment, Sequences};
use crate::alphabets::ParsimonySet;
use crate::evolutionary_models::EvoModel;
use crate::tree::Tree;
use crate::tree::{NodeIdx::Internal as Int, NodeIdx::Leaf};

pub mod costs;
use costs::*;
pub mod matrices;
pub(crate) use matrices::*;

pub(crate) type CostMatrix = DMatrix<f64>;

#[derive(Clone, Debug, PartialEq)]
pub struct Rounding {
    pub round: bool,
    pub digits: usize,
}

impl Rounding {
    pub fn zero() -> Self {
        Rounding {
            round: true,
            digits: 0,
        }
    }
    pub fn four() -> Self {
        Rounding {
            round: true,
            digits: 4,
        }
    }
    pub fn none() -> Self {
        Rounding {
            round: false,
            digits: 0,
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct Zero {
    is_set: bool,
}

impl Zero {
    pub fn yes() -> Self {
        Zero { is_set: true }
    }
    pub fn no() -> Self {
        Zero { is_set: false }
    }
    pub fn is_set(&self) -> bool {
        self.is_set
    }
}

pub trait ParsimonyModel: Display + EvoModel {
    fn scoring_matrix(&self, time: f64, rounding: Rounding) -> (CostMatrix, f64);

    fn scoring_matrix_corrected(
        &self,
        time: f64,
        diagonals: Zero,
        rounding: Rounding,
    ) -> (CostMatrix, f64);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Direction {
    Matc,
    GapInY,
    GapInX,
}

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
    x_blen: f64,
    y_info: &[ParsimonySite],
    y_blen: f64,
    scoring: &dyn ParsimonyCosts,
    rng: fn(usize) -> usize,
) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
    let mut pars_mats =
        ParsimonyAlignmentMatrices::new(x_info, x_blen, y_info, y_blen, scoring, rng);

    pars_mats.fill_matrices();
    pars_mats.traceback()
}

fn pars_align(
    x_info: &[ParsimonySite],
    x_blen: f64,
    y_info: &[ParsimonySite],
    y_blen: f64,
    scoring: &dyn ParsimonyCosts,
) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
    pars_align_w_rng(x_info, x_blen, y_info, y_blen, scoring, rng_len)
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
                let (info, alignment, score) =
                    pars_align(x_info, x_branch, y_info, y_branch, scoring);
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
