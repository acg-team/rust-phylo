use std::cell::RefCell;
use std::fmt::Display;

use approx::assert_relative_eq;
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use lazy_static::lazy_static;
use nalgebra::{DMatrix, DVector};

use crate::alignment::{AncestralAlignment, Mapping};
use crate::likelihood::{ModelSearchCost, ParamRange, TreeSearchCost};
use crate::phylo_info::PhyloInfo;
use crate::random::FakeGenerator;
use crate::substitution_models::FreqVector;
use crate::tkf_model::reestimate::EdgeSeqsReestimator;
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::tree::Tree;
use crate::{bail, Result, REPORT_ISSUES_URL};

lazy_static! {
    pub(super) static ref DUMMY_FREQS: DVector<f64> = DVector::<f64>::zeros(0);
}

pub(super) static DEFAULT_LAMBDA: f64 = 1.0;
pub(super) static DEFAULT_MU: f64 = 1.1;
pub(super) static DEFAULT_LAMBDA_MU_RATIO: f64 = 0.9;
pub(super) static DEFAULT_R: f64 = 0.5;

/// Events that can happen on a branch in the TKF model.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum Event {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(super) enum NumBlockAppearances {
    Variable(usize),
    Fixed,
}

/// All the [blocks](`Block`) in an [alignment](`AncestralAlignment`).
pub type Blocks = Vec<Block>;

/// A block is a contiguous segment of the alignment where the presence or absence of characters in
/// the ancestral mappings is uniform within every sequence.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Block {
    /// The right exclusive interval border of the block.
    /// For example, if the block is [3, 5), the block border is 5.
    border: usize,
    /// Since within a block the presence or absence of characters in the ancestral mappings are the same,
    /// we can just use one representative site to determine the [event](`Event`) for the whole block.
    rep_site: usize,
    /// The length of the block, i.e., border - previous block border.
    len: usize,
    /// Either the number of times this block's border appears in the alignment, or whether it should be
    /// treated as a fixed block independently of the number of appearances.
    /// Under the [`crate::tkf_model::TKF91IndelModel`] all blocks are
    /// [fixed](`NumBlockAppearances::Fixed`), since it's a single site model.
    /// Under the [`crate::tkf_model::TKF92IndelModel`] the blocks are
    /// [variable](`NumBlockAppearances::Variable`), since during
    /// [re-estimation](`EdgeSeqsReestimator`) the block borders can change and we need to keep
    /// track of how many times they appear in the alignment to know when to merge blocks.
    num_appearances: NumBlockAppearances,
}

impl Block {
    /// Creates a new [`Block`], panicking if any invariant is violated:
    /// - `border > 0`
    /// - `len > 0`
    /// - `rep_site` is within the half-open interval `[border - len, border)`
    pub(super) fn new(
        border: usize,
        rep_site: usize,
        len: usize,
        num_appearances: NumBlockAppearances,
    ) -> Self {
        assert!(border > 0, "Block border must be greater than 0.");
        assert!(len > 0, "Block length must be greater than 0.");
        assert!(
            rep_site >= border - len && rep_site < border,
            "Block rep_site {rep_site} must lie within [{}, {border}).",
            border - len
        );
        Block {
            border,
            rep_site,
            len,
            num_appearances,
        }
    }

    pub(super) fn rep_site(&self) -> usize {
        self.rep_site
    }

    /// Returns a mutable reference to the number of appearances of this block's border (i.e., [`Self::coordinates`]\().1) in the alignment.
    pub(super) fn num_appearances_mut(&mut self) -> &mut NumBlockAppearances {
        &mut self.num_appearances
    }

    #[cfg(test)]
    pub(super) fn num_appearances(&self) -> NumBlockAppearances {
        self.num_appearances
    }

    /// Returns the coordinates of the block as a tuple [start, end)
    pub fn coordinates(&self) -> (usize, usize) {
        let start = self.border - self.len;
        let end = self.border;
        (start, end)
    }

    /// Returns the length of the block (which is always greater than 0).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Trait for TKF indel models (i.e., [TKF91IndelModel](`crate::tkf_model::TKF91IndelModel`),
/// [TKF92IndelModel](`crate::tkf_model::TKF92IndelModel`)).
#[allow(clippy::upper_case_acronyms)]
pub trait TKFModel: Clone + Display {
    // TODO: it might be better for model optimisation to have parameter lambda and scale s = mu/lambda,
    // because of the constraint that mu > lambda.
    // See issue #152 https://github.com/acg-team/rust-phylo/issues/152
    fn lambda(&self) -> f64;
    fn mu(&self) -> f64;
    /// [TKF91](crate::tkf_model::tkf91) has 2 parameters: `lambda` and `mu`, [TKF92](crate::tkf_model::tkf92)
    /// has 3 parameters: `lambda`, `mu` and `r`.
    /// The parameter `r` in [TKF92](crate::tkf_model::tkf92) is used to model the length distribution of inserted segments,
    /// i.e., in [`super::TKF92IndelModel::insertion_factor_at_non_root`] and
    /// [`super::TKF92IndelModel::insertion_factor_at_root`].
    fn params(&self) -> &[f64];
    fn set_param(&mut self, idx: usize, value: f64);
    fn param_range(&self, idx: usize) -> ParamRange;
    /// Returns the factor corresponding to an insertion event at the root.
    fn insertion_factor_at_root(&self) -> f64;
    /// Returns the factor corresponding to an insertion event at a non-root node.
    fn insertion_factor_at_non_root(&self, beta: f64) -> f64;
    /// Given the subtree event factor for the root (i.e., the tree event factor)
    /// and the [block length](`Block::len`), returns the log probability of the [block](`TKFModel::get_blocks`) under the model.
    fn block_prob(&self, tree_event_factor: f64, block_len: usize) -> f64;
    /// For every [block](`Block`) (i.e., an alignment slice) as determined by this method and factors
    /// corresponding to the evolutionary events in this block [`TKFModel::block_prob`] computes
    /// the log probability of the block under the model.
    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Blocks;
}

// TODO: link our paper once it is published. For now see original TKF92 paper: https://doi.org/10.1007/bf00163848
/// This struct holds intermediate values for the computation of the log likelihood
/// of an ancestral alignment and tree under a TKF indel model, i.e., without substitutions.
/// The intermediate values are needed for re-alignment, which is not implemented yet.
/// See issue #150 https://github.com/acg-team/rust-phylo/issues/150
#[derive(Clone, Debug)]
pub(super) struct TKFIndelModelInfo {
    /// node_event_factor[(node, block)] = the probability factor for the event
    /// on the edge above <node> for the block with id <block>.
    /// See [`TKFIndelCost`] and
    /// [`TKFIndelCost::event_factor`].
    pub(super) node_event_factor: DMatrix<f64>,
    /// subtree_event_factor[(node, block)] = the product of the event probability factors
    /// for all edges in the subtree rooted in <node> for the block with id <block>,
    /// including the edge above <node>.
    /// See [`TKFIndelCost::set_node_values`].
    pub(super) subtree_event_factor: DMatrix<f64>,

    /// node_eta[(node, block)] = node_eta[(node, block)] = eta if the current event is an
    /// insertion and the previous one was a deletion, 0 otherwise.
    /// See [`TKFIndelCost::eta_for_non_root`].
    pub(super) node_eta: DMatrix<f64>,
    /// subtree_eta[(node, block)] = sum of node_eta for all nodes in the subtree rooted in <node>
    /// for the block with id <block>. Since we only have one insertion per column, at most one
    /// node in the subtree can contribute to this sum.
    pub(super) subtree_eta: DMatrix<f64>,

    /// beta[node] = beta(node.blen)), precomputed for each node.
    /// See [`beta`] function.
    pub(super) beta: Vec<f64>,
    /// n0[node] = n0(node.blen), precomputed for each node.
    /// See [`n0`] function.
    pub(super) n0: Vec<f64>,
    /// h1[node] = h1(node.blen), precomputed for each node.
    /// See [`h1`] function.
    pub(super) h1: Vec<f64>,
    /// insertion[node], precomputed for each node.
    /// See [`TKFModel::insertion_factor_at_root`] and [`TKFModel::insertion_factor_at_non_root`].
    pub(super) insertion: Vec<f64>,
    /// eta[node] = n1/ (n0 * lambda * beta(node.blen)), precomputed for each node.
    /// See [`eta`] function.
    pub(super) eta: Vec<f64>,

    /// The blocks in the alignment, which are determined by
    /// [`TKFModel::get_blocks`]
    pub(super) blocks: Blocks,

    /// previous_event_deletion[node] = true if the last event was a deletion for a that <node>.
    /// See [`TKFIndelCost::determine_event`] and [`TKFIndelCost::update_previous_event`].
    pub(super) previous_event_deletion: FixedBitSet,

    /// valid[node] = true if the intermediate values for that <node> are valid.
    pub(super) valid: FixedBitSet,
    /// valid_for_reestimation[node] = true if the intermediate values can be used for re-estimation.
    /// Since for re-estimation we don't need the subtree values for the internal nodes except the
    /// root. So if many re-estimations are done for a fixed tree and model, we can save time by not
    /// recomputing subtree values for internal nodes that are not the root.
    pub(super) valid_for_reestimation: FixedBitSet,
}

impl TKFIndelModelInfo {
    pub(super) fn new<AA: AncestralAlignment, T: TKFModel>(
        model: &T,
        phylo: &PhyloInfo<AA>,
    ) -> TKFIndelModelInfo {
        let blocks = model.get_blocks(&phylo.msa);
        let n_blocks = blocks.len();
        let n_nodes = phylo.tree.len();
        TKFIndelModelInfo {
            node_event_factor: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            subtree_event_factor: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_eta: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            subtree_eta: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            beta: vec![0.0; n_nodes],
            n0: vec![0.0; n_nodes],
            h1: vec![0.0; n_nodes],
            insertion: vec![0.0; n_nodes],
            eta: vec![0.0; n_nodes],
            blocks,
            previous_event_deletion: FixedBitSet::with_capacity(n_nodes),
            valid: FixedBitSet::with_capacity(n_nodes),
            valid_for_reestimation: FixedBitSet::with_capacity(n_nodes),
        }
    }
}

/// Computes the log likelihood of an [ancestral alignment](`AncestralAlignment`)
/// and tree under a [TKF](`TKFModel`) indel model, i.e., without substitutions.
#[derive(Debug)]
pub struct TKFIndelCost<T: TKFModel, AA: AncestralAlignment> {
    pub(super) model: T,
    pub(super) phylo: PhyloInfo<AA>,
    pub(super) model_info: RefCell<TKFIndelModelInfo>,
}

impl<T: TKFModel, AA: AncestralAlignment + Clone> Clone for TKFIndelCost<T, AA> {
    fn clone(&self) -> Self {
        TKFIndelCost {
            model: self.model.clone(),
            phylo: self.phylo.clone(),
            model_info: RefCell::new(self.model_info.borrow().clone()),
        }
    }
}

impl<T: TKFModel, AA: AncestralAlignment> Display for TKFIndelCost<T, AA> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.model)
    }
}

impl<T: TKFModel, AA: AncestralAlignment> TKFIndelCost<T, AA> {
    pub(super) fn set_all_nodes(&self) {
        for node_idx in self.phylo.tree.postorder() {
            match node_idx {
                Internal(_) => {
                    if self.phylo.tree.root == *node_idx {
                        self.set_root();
                    } else {
                        self.set_non_root(node_idx);
                    }
                }
                Leaf(_) => {
                    self.set_non_root(node_idx);
                }
            };
        }
    }

    pub(super) fn logl(&self) -> f64 {
        self.set_all_nodes();
        self.logl_from_root_model_info()
    }

    pub(super) fn logl_from_root_model_info(&self) -> f64 {
        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let root_id = usize::from(self.phylo.tree.root);
        let mut logl = (1.0 - lambda / mu).ln();
        let model_info = self.model_info.borrow();
        // for every node except the root
        for node in self.phylo.tree.preorder().iter().skip(1) {
            logl += log_i1(lambda, model_info.beta[usize::from(node)]);
        }
        for (block_id, block) in model_info.blocks.iter().enumerate() {
            logl += model_info.subtree_eta[(root_id, block_id)];
            let tree_event_factor = model_info.subtree_event_factor[(root_id, block_id)];
            logl += self.model.block_prob(tree_event_factor, block.len);
        }
        logl
    }

    fn set_root(&self) {
        let root_idx = &self.phylo.tree.root;
        let root_id = usize::from(root_idx);
        if self.model_info.borrow().valid[root_id] {
            return;
        }
        self.reset_cache(root_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let event = self.determine_event(root_idx, block_id);
            let node_event_factor = self.event_factor(root_idx, event);
            let node_eta = 0.0;
            self.set_node_values(root_idx, block_id, node_event_factor, node_eta);
        }
        let mut model_info = self.model_info.borrow_mut();
        model_info.valid.set(root_id, true);
        model_info.valid_for_reestimation.set(root_id, true);
    }

    fn set_non_root(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        if self.model_info.borrow().valid[node_id] {
            return;
        }
        self.reset_cache(node_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            if block_id == 0 {
                self.model_info
                    .borrow_mut()
                    .previous_event_deletion
                    .set(usize::from(node_idx), false);
            }
            let event = self.determine_event(node_idx, block_id);
            let node_event_factor = self.event_factor(node_idx, event);
            let node_eta = self.eta_for_non_root(node_idx, event);
            self.set_node_values(node_idx, block_id, node_event_factor, node_eta);
            if let Some(val) = self.updated_previous_is_deletion(event) {
                self.model_info
                    .borrow_mut()
                    .previous_event_deletion
                    .set(usize::from(node_idx), val);
            }
        }
        let mut model_info = self.model_info.borrow_mut();
        if let Some(parent_idx) = self.phylo.tree.parent(node_idx) {
            model_info.valid.set(usize::from(parent_idx), false);
        }
        model_info.valid.set(node_id, true);
        model_info.valid_for_reestimation.set(node_id, true);
    }

    pub(super) fn updated_previous_is_deletion(&self, event: Event) -> Option<bool> {
        match event {
            Event::Deletion => Some(true),
            Event::Insertion | Event::Homolog => Some(false),
            Event::Nothing => None,
        }
    }

    pub(super) fn reset_cache(&self, node_idx: &NodeIdx) {
        let node_id = usize::from(node_idx);
        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let blen = self.phylo.tree.node(node_idx).blen;
        let beta = beta(lambda, mu, blen);
        let mut model_info = self.model_info.borrow_mut();
        model_info.beta[node_id] = beta;
        model_info.n0[node_id] = n0(mu, beta);
        model_info.h1[node_id] = h1(lambda, mu, beta, blen);
        model_info.insertion[node_id] = if node_idx == &self.phylo.tree.root {
            self.model.insertion_factor_at_root()
        } else {
            self.model.insertion_factor_at_non_root(beta)
        };
        model_info.previous_event_deletion.set(node_id, false);
        model_info.eta[node_id] = eta(lambda, mu, beta, model_info.n0[node_id], blen);
        model_info.valid.set(node_id, false);
    }

    fn set_node_values(
        &self,
        node_idx: &NodeIdx,
        block_id: usize,
        node_event_factor: f64,
        node_eta: f64,
    ) {
        let node_id = usize::from(node_idx);
        let mut model_info = self.model_info.borrow_mut();
        model_info.node_event_factor[(node_id, block_id)] = node_event_factor;
        model_info.node_eta[(node_id, block_id)] = node_eta;
        let mut substree_event_factor = node_event_factor;
        let mut subtree_eta = node_eta;
        for child in &self.phylo.tree.node(node_idx).children {
            let child_id = usize::from(child);
            substree_event_factor *= model_info.subtree_event_factor[(child_id, block_id)];
            subtree_eta += model_info.subtree_eta[(child_id, block_id)];
        }
        model_info.subtree_event_factor[(node_id, block_id)] = substree_event_factor;
        model_info.subtree_eta[(node_id, block_id)] = subtree_eta;
    }

    /// Determines the event that happened on the edge above `node_idx` for the given `block_id`
    /// based on the [ancestral alignment](`AncestralAlignment`).
    pub(super) fn determine_event(&self, node_idx: &NodeIdx, block_id: usize) -> Event {
        // the presence or absence of characters is the same for all sites in a block
        // so we can just check the last site of the block
        let site = self.model_info.borrow().blocks[block_id].rep_site;

        let parent_is_gap = match self.phylo.tree.node(node_idx).parent {
            Some(parent_idx) => match parent_idx {
                Internal(_) => self.phylo.msa.ancestral_map(&parent_idx)[site].is_none(),
                _ => unreachable!("The parent of a node cannot be a leaf. Please report this at {REPORT_ISSUES_URL}"),
            },
            // root has no parent, so we treat the position as a gap, then if there is
            // a character at the root the event will be determined as an insertion,
            // which is correct under the TKF model
            None => true,
        };

        let current_is_gap = match node_idx {
            Internal(_) => self.phylo.msa.ancestral_map(node_idx)[site].is_none(),
            Leaf(_) => self.phylo.msa.leaf_map(node_idx)[site].is_none(),
        };
        if !parent_is_gap && current_is_gap {
            Event::Deletion
        } else if !parent_is_gap && !current_is_gap {
            Event::Homolog
        } else if parent_is_gap && !current_is_gap {
            Event::Insertion
        } else {
            Event::Nothing
        }
    }

    pub(super) fn event_factor(&self, node_idx: &NodeIdx, event: Event) -> f64 {
        let node_id = usize::from(node_idx);
        match event {
            Event::Deletion => self.model_info.borrow().n0[node_id],
            Event::Homolog => self.model_info.borrow().h1[node_id],
            Event::Insertion => self.model_info.borrow().insertion[node_id],
            Event::Nothing => 1.0,
        }
    }

    /// Returns eta if the current event is an insertion and the previous one was a deletion, 0 otherwise.
    /// See [`eta`] function.
    /// Since there can't be a deletion at the root (it has no parent), this function is only for non-root nodes.
    pub(super) fn eta_for_non_root(&self, node_idx: &NodeIdx, event: Event) -> f64 {
        let model_info = self.model_info.borrow();
        if matches!(event, Event::Insertion)
            && model_info.previous_event_deletion[usize::from(node_idx)]
        {
            model_info.eta[usize::from(node_idx)]
        } else {
            0.0
        }
    }

    /// Checks whether the new mapping conforms to the current blocking of the alignment, i.e.,
    /// presence or absence of characters in the ancestral mapping is uniform within every block.
    /// If this is not the case, an error is returned.
    fn mapping_conforms_to_blocking(&self, mapping: &Mapping) -> Result<()> {
        let blocks = &self.model_info.borrow().blocks;
        for block in blocks {
            let (start, end) = block.coordinates();
            let mapping_slice = &mapping[start..end];
            if let Some((first, rest)) = mapping_slice.split_first() {
                let required_state = first.is_some();
                if !rest.iter().all(|x| x.is_some() == required_state) {
                    bail!(
                    TKF,
                    "the new mapping does not conform to the current blocking of the alignment, \
                     i.e., presence or absence of characters in the ancestral mapping \
                     is not uniform within every block."
                );
                }
            }
        }
        Ok(())
    }

    /// Given the new ancestral mapping and comparing it to the old one, checks whether
    /// the block border at `block_id` is still enforced by the alignment. It does this by
    /// decreasing the count of appearances for this block border and if it reaches zero, i.e.,
    /// the alignment no longer enforces this block border, it returns true, which is a signal to
    /// merge the two blocks around this block border, see [`Self::update_mappings_and_model_info`].
    /// Since we are only checking sites at block borders, we can only merge blocks and not split them.
    fn decrease_count_and_is_zero(&self, old: &Mapping, new: &Mapping, block_id: usize) -> bool {
        let blocks = &mut self.model_info.borrow_mut().blocks;
        let prev_site = blocks[block_id - 1].rep_site;
        let curr_site = blocks[block_id].rep_site;
        let transition_in_old = old[prev_site].is_some() ^ old[curr_site].is_some();
        let transition_in_new = new[prev_site].is_some() ^ new[curr_site].is_some();

        if transition_in_old && !transition_in_new {
            match &mut blocks[block_id - 1].num_appearances_mut() {
                NumBlockAppearances::Variable(count) => {
                    assert!(
                        *count > 0,
                        "Tried to subtract one from already zero count of block appearances. \
                         Please report this at {REPORT_ISSUES_URL}"
                    );
                    *count -= 1;
                    return *count == 0;
                }
                NumBlockAppearances::Fixed => {
                    return false;
                }
            }
        }
        false
    }

    /// Given a new ancestral mapping for a node updates the mapping in the alignment and updates
    /// the model info accordingly, i.e., merges blocks if necessary and adjusts model info matrix
    /// dimensions in that case and sets the tmp values for the node and its children as invalid,
    /// since the [events](Event) on these edges might have changed due to the updated mapping.
    ///
    /// # Errors
    /// - bails if the length of the new mapping does not match the alignment length
    /// - bails if the node is not an internal one of the [`PhyloInfo`]s tree
    /// - bails if the new mapping does not [conform to the current blocking](`Self::mapping_conforms_to_blocking`) of the alignment
    ///
    /// # Notes
    /// If all initial ancestral mappings do not [enforce any block borders](TKFModel::get_blocks)
    /// that are not already enforced by the leaf mappings, this method will never merge blocks and
    /// therefore never change the dimensions of the [`TKFIndelModelInfo`] matrices, since the
    /// [NumBlockAppearances] will never [reach zero](`Self::decrease_count_and_is_zero`).
    /// So, in the case where the ancestral mappings are estimated with column i.i.d. methods
    /// from the leaf alignment, this method will not merge blocks.
    /// However, if the ancestral mappings are taken from simulation for example,
    /// then ancestral mappings might enforce block borders that are not enforced by the leaf mappings,
    /// and then during re-estimation these blocks might need to be merged if the new ancestral mapping
    /// no longer enforces the block border.
    pub(super) fn update_mappings_and_model_info(
        &mut self,
        node_idx: &NodeIdx,
        new_map: Mapping,
    ) -> Result<()> {
        if self.phylo.msa.ancestral_maps().get(node_idx).is_none() {
            match node_idx {
                Internal(_) => bail!(AncestralAlignment, "no ancestral map found for: {node_idx}"),
                Leaf(_) => bail!(
                    AncestralAlignment,
                    "ancestral map cannot be set for a leaf node like {node_idx}"
                ),
            }
        }
        if new_map.len() != self.phylo.msa.len() {
            bail!(
                AncestralAlignment,
                "mapping length {} does not match MSA length {}",
                new_map.len(),
                self.phylo.msa.len()
            );
        }
        self.mapping_conforms_to_blocking(&new_map)?;
        self.update_mappings_and_model_info_unchecked(node_idx, new_map);
        Ok(())
    }

    /// This is the unchecked version of [`Self::update_mappings_and_model_info`].
    pub(super) fn update_mappings_and_model_info_unchecked(
        &mut self,
        node_idx: &NodeIdx,
        new_map: Mapping,
    ) {
        let prev_map = self.phylo.msa.ancestral_map(node_idx).clone();
        self.phylo
            .msa
            .update_ancestral_map_unchecked(node_idx, new_map);

        let mut block_id = 1;
        let mut num_blocks = self.model_info.borrow().blocks.len();
        let mut merged = false;
        while block_id < num_blocks {
            let new_map = self.phylo.msa.ancestral_map(node_idx);
            let merge_blocks = self.decrease_count_and_is_zero(&prev_map, new_map, block_id);
            if merge_blocks {
                self.merge_blocks_and_adjust_dimensions(block_id);
                merged = true;
                num_blocks -= 1;
            } else {
                block_id += 1;
            }
        }
        if merged {
            // the node_eta of all nodes need to be correctly updated next time the logl is called
            self.model_info.borrow_mut().valid.clear();
        } else {
            // setting the node and its children tmp values as invalid, since the events on these edges might have
            // changed due to the updated mapping
            self.set_affected_nodes_as_invalid(node_idx);
        }
    }

    fn merge_blocks_and_adjust_dimensions(&mut self, block_id: usize) {
        let mut model_info = self.model_info.borrow_mut();
        // updating the blocks in the model_info, i.e., merge the previous block with the current one
        let additional_len = model_info.blocks[block_id - 1].len;
        model_info.blocks.remove(block_id - 1);
        model_info.blocks[block_id - 1].len += additional_len;

        // updating the dimensions of the model_info matrices
        // model_info.node_event_factor.remove_column(block_id - 1); does not work
        // since: cannot move out of dereference of `std::cell::RefMut<'_, tkf_model::tkf_indel::TKFIndelModelInfo>`
        // so instead we do this mem trick
        let remove_col = |matrix: &mut DMatrix<f64>| {
            let old = std::mem::replace(matrix, DMatrix::zeros(0, 0));
            *matrix = old.remove_column(block_id - 1);
        };
        remove_col(&mut model_info.node_event_factor);
        remove_col(&mut model_info.subtree_event_factor);
        remove_col(&mut model_info.node_eta);
        remove_col(&mut model_info.subtree_eta);
    }

    fn set_affected_nodes_as_invalid(&self, node_idx: &NodeIdx) {
        let mut model_info = self.model_info.borrow_mut();
        model_info.valid.set(usize::from(node_idx), false);
        for child in &self.phylo.tree.node(node_idx).children {
            model_info.valid.set(usize::from(child), false);
        }
    }
}

impl<T: TKFModel, AA: AncestralAlignment> ModelSearchCost for TKFIndelCost<T, AA> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn param_count(&self) -> usize {
        self.model.params().len()
    }

    fn param(&self, idx: usize) -> f64 {
        self.model.params()[idx]
    }

    fn set_param(&mut self, idx: usize, value: f64) {
        self.model.set_param(idx, value);
        self.model_info.borrow_mut().valid.clear();
    }

    /// Returns the valid range for a model parameter [min, max], inclusive.
    /// Assumes that current parameter values are valid.
    fn param_range(&self, idx: usize) -> ParamRange {
        self.model.param_range(idx)
    }

    fn set_freqs(&mut self, _: FreqVector) {}

    fn empirical_freqs(&self) -> FreqVector {
        // At the time of writing this, this method is only used to set the frequencies of
        // the model, but the TKF92IndelCost does not have frequencies.
        self.phylo.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        &DUMMY_FREQS
    }
}

impl<T: TKFModel, AA: AncestralAlignment> TreeSearchCost for TKFIndelCost<T, AA> {
    fn cost(&self) -> f64 {
        self.logl()
    }

    fn update_tree(&mut self, tree: Tree) {
        let mut dirty_nodes = vec![];
        let mut model_info = self.model_info.borrow_mut();
        for idx in tree.dirty.ones() {
            model_info.valid.set(idx, false);
            model_info.valid_for_reestimation.set(idx, false);
            dirty_nodes.push(idx);
        }
        drop(model_info);

        let update_due_to_nni = dirty_nodes.len() == 1 && {
            // check if children of the dirty node are different than before
            let mut previous_children = self.phylo.tree.nodes[dirty_nodes[0]]
                .children
                .iter()
                .collect_vec();
            let mut new_children = tree.nodes[dirty_nodes[0]].children.iter().collect_vec();
            previous_children.sort();
            new_children.sort();

            previous_children != new_children
        };
        self.phylo.tree = tree;
        if update_due_to_nni {
            let v2 = self.tree().nodes[dirty_nodes[0]].idx;
            // TODO: For now we use the FakeGenerator to have deterministic behavior, but in the future
            // we should use a proper RNG here. But pass the RNG from outside and not
            // create a new one each time, see issue #142 https://github.com/acg-team/rust-phylo/issues/142
            let rng = &mut FakeGenerator::default();
            let mut reestimator = EdgeSeqsReestimator::new(self, rng);
            let dp_logl = reestimator.reestimate_unchecked(&v2);
            assert_relative_eq!(dp_logl, self.logl(), epsilon = 1e-10);
        }
        self.phylo.tree.clean();
    }

    fn tree(&self) -> &Tree {
        &self.phylo.tree
    }
}

/// Returns the value of `beta(t)` for a branch of length/time `t`.
/// It is called beta(t) in the TKF papers.
#[inline]
pub(super) fn beta(lambda: f64, mu: f64, time: f64) -> f64 {
    let exp_term = ((lambda - mu) * time).exp();
    (1.0 - exp_term) / (mu - lambda * exp_term)
}

/// Returns the log probability factor of a character being inserted to the right of the immortal link
/// along a branch of length `time`, i.e., at the very left of the sequence.
/// The `time` is also implicitly included in `beta`.
/// It is called `p''_1` in the TKF papers.
#[inline]
pub(super) fn log_i1(lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta).ln()
}

/// Returns the probability factor of a homologous character surviving along a branch of length `time`.
/// The `time` is also implicitly included in `beta`.
/// It is called `p_1` in the TKF papers.
#[inline]
pub(super) fn h1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    (-mu * time).exp() * (1.0 - lambda * beta)
}

/// Returns the probability factor of a character being deleted along a branch of length `time`.
/// It is called `p'_0` in the TKF papers.
/// The `time` is implicitly included in `beta`.
#[inline]
pub(super) fn n0(mu: f64, beta: f64) -> f64 {
    mu * beta
}

/// Returns the log probability factor of a new character being inserted right of a character that is
/// deleted along a branch of length `time`.
/// The `time` is also implicitly included in beta.
/// It is called `p'_1` in the TKF papers.
#[inline]
pub(super) fn log_n1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    ((1.0 - (-mu * time).exp() - mu * beta) * (1.0 - lambda * beta)).ln()
}

/// Returns the log of the `n1 / (n0 * lambda * beta)`.
/// This is used in the case where an insertion follows a deletion,
/// since the event factors included `n0` for the deletion and `lambda * beta` for the insertion
/// but under the TKF model they are not independent and instead `n1` should be used.
/// `Eta` corrects for that.
/// The `time` is also implicitly included in `beta` and `n0`.
#[inline]
pub(super) fn eta(lambda: f64, mu: f64, beta: f64, n0: f64, time: f64) -> f64 {
    let mut eta = log_n1(lambda, mu, beta, time);
    eta -= (lambda * beta).ln();
    eta -= n0.ln();
    eta
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {

    use super::*;
    use crate::alphabets::Alphabet;
    use crate::tkf_model::tests::setup_test_phylo;
    use crate::tkf_model::TKF91IndelCostBuilder;
    use crate::tkf_model::TKF92IndelCostBuilder;
    use crate::tkf_model::TKFModel;

    #[cfg(test)]
    fn validate_lambda_mu(l: f64, m: f64, l_expected: f64, m_expected: f64) {
        let cost = TKF91IndelCostBuilder::new(l, m, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
        assert_eq!(cost.model.lambda(), l_expected);
        assert_eq!(cost.model.mu(), m_expected);
        let cost = TKF92IndelCostBuilder::new(l, m, 0.1, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
        assert_eq!(cost.model.lambda(), l_expected);
        assert_eq!(cost.model.mu(), m_expected);
    }

    #[cfg(test)]
    fn validate_r(r: f64, r_expected: f64) {
        let cost = TKF92IndelCostBuilder::new(1.0, 2.0, r, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
        assert_eq!(cost.model.r(), r_expected);
    }

    #[test]
    fn tkf_validate_params_for_builder() {
        validate_lambda_mu(-1.0, -2.0, DEFAULT_LAMBDA, DEFAULT_MU);
        validate_lambda_mu(0.0, 2.0, DEFAULT_LAMBDA_MU_RATIO * 2.0, 2.0);
        validate_lambda_mu(2.0, -0.1, 2.0, 2.0 / DEFAULT_LAMBDA_MU_RATIO);
        validate_lambda_mu(2.0, 1.9999, 2.0, 2.0 / DEFAULT_LAMBDA_MU_RATIO);
        validate_lambda_mu(1.2, 1.21, 1.2, 1.21);
        validate_r(-0.5, DEFAULT_R);
        validate_r(0.0, DEFAULT_R);
        validate_r(1.0, DEFAULT_R);
        validate_r(1.5, DEFAULT_R);
        validate_r(0.1, 0.1);
    }
}
