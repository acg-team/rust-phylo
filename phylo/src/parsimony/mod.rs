use std::default;
use std::fmt::Display;

use hashbrown::HashMap;
use log::info;
use nalgebra::DMatrix;

use crate::alignment::{Aligner, Alignment, InternalMapping, PairwiseAlignment, Sequences};
use crate::evolutionary_models::EvoModel;
use crate::tree::{NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::Result;

pub mod scoring;
use scoring::*;
pub mod matrices;
pub(crate) use matrices::*;
pub(crate) mod helpers;
pub(crate) use helpers::*;

pub(crate) type CostMatrix = DMatrix<f64>;

pub trait ParsimonyModel: Display + EvoModel {
    fn scoring(&self, time: f64, diagonals: &DiagonalZeros, rounding: &Rounding) -> CostMatrix;
}

pub struct ParsimonyAligner<PS: ParsimonyScoring> {
    pub scoring: PS,
}

impl<PS: ParsimonyScoring + Clone> Aligner for ParsimonyAligner<PS> {
    fn align(&self, seqs: &Sequences, tree: &Tree) -> Result<Alignment> {
        self.align_with_scores(seqs, tree).map(|(a, _)| a)
    }
}

impl default::Default for ParsimonyAligner<SimpleScoring> {
    fn default() -> Self {
        ParsimonyAligner {
            scoring: SimpleScoring::new(1.0, GapCost::new(2.5, 0.5)),
        }
    }
}

impl<'a, PS: ParsimonyScoring + Clone> ParsimonyAligner<PS> {
    pub fn new(scoring: PS) -> ParsimonyAligner<PS> {
        ParsimonyAligner { scoring }
    }

    pub fn align_with_scores(
        &self,
        seqs: &'a Sequences,
        tree: &'a Tree,
    ) -> Result<(Alignment, Vec<f64>)> {
        info!("Starting the IndelMAP alignment.");

        let order = tree.postorder();

        let mut node_info = vec![Vec::<ParsimonySite>::new(); tree.len()];
        let mut alignments = InternalMapping::with_capacity(tree.internals().len());
        let mut scores = vec![0.0; tree.len()];

        for &node_idx in order {
            info!("Processing {}{}.", node_idx, tree.node(&node_idx).id);
            match node_idx {
                Int(idx) => {
                    let child = tree.children(&node_idx);
                    let x_idx = child[0];
                    let y_idx = child[1];

                    let x_info = &node_info[usize::from(child[0])];
                    let x_blen = tree.node(&child[0]).blen;

                    let y_info = &node_info[usize::from(child[1])];
                    let y_blen = tree.node(&child[1]).blen;

                    info!("Aligning sequences at nodes {} and {}", x_idx, y_idx,);
                    let (info, alignment, score) =
                        self.pairwise_align(x_info, x_blen, y_info, y_blen, rng_len);
                    node_info[idx] = info;
                    alignments.insert(node_idx, alignment);
                    scores[idx] = score;
                    info!("Alignment complete with score {}.\n", score);
                }
                Leaf(idx) => {
                    let pars_sets = seqs
                        .record_by_id(tree.node_id(&node_idx))
                        .seq()
                        .iter()
                        .map(|c| seqs.alphabet().parsimony_set(c))
                        .collect::<Vec<_>>();
                    node_info[idx] = pars_sets.into_iter().map(ParsimonySite::leaf).collect();
                    info!("Processed leaf node.\n");
                }
            }
        }
        info!("Finished IndelMAP alignment.");

        let mut alignment = Alignment {
            seqs: seqs.into_gapless(),
            leaf_map: HashMap::new(),
            node_map: alignments,
            leaf_encoding: HashMap::new(),
        };
        let leaf_map = alignment.compile_leaf_map(&tree.root, tree).unwrap();
        alignment.leaf_map = leaf_map;
        alignment.leaf_encoding = alignment.seqs.generate_leaf_encoding();
        Ok((alignment, scores))
    }

    fn pairwise_align(
        &self,
        x_info: &[ParsimonySite],
        x_blen: f64,
        y_info: &[ParsimonySite],
        y_blen: f64,
        rng: fn(usize) -> usize,
    ) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
        let mut pars_mats =
            ParsimonyAlignmentMatrices::new(x_info, x_blen, y_info, y_blen, &self.scoring, rng);

        pars_mats.fill_matrices();
        pars_mats.traceback()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
