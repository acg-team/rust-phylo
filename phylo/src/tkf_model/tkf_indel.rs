use std::cell::RefCell;
use std::fmt::Display;

use fixedbitset::FixedBitSet;
use lazy_static::lazy_static;
use nalgebra::{DMatrix, DVector};

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ModelSearchCost, ParamRange};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::FreqVector;
use crate::tree::NodeIdx::{self, Internal, Leaf};

lazy_static! {
    pub(super) static ref DUMMY_FREQS: DVector<f64> = DVector::<f64>::zeros(0);
}

pub(super) static DEFAULT_LAMBDA: f64 = 1.0;
pub(super) static DEFAULT_MU: f64 = 1.1;
pub(super) static DEFAULT_LAMBDA_MU_RATIO: f64 = 0.9;
pub(super) static DEFAULT_R: f64 = 0.5;

#[derive(Copy, Clone)]
enum Event {
    Insertion,
    Deletion,
    Homolog,
    Nothing,
}

/// Trait for TKF indel models (i.e., [TKF91](crate::tkf_model::tkf91) and
/// [TKF92](crate::tkf_model::tkf92)).
#[allow(clippy::upper_case_acronyms)]
pub trait TKFModel: Clone + Display {
    // TODO: it might be better for model optimisation to have parameter lambda and scale s = mu/lambda,
    // because of the constraint that mu > lambda.
    fn lambda(&self) -> f64;
    fn mu(&self) -> f64;
    /// [TKF91](crate::tkf_model::tkf91) has 2 parameters: lambda and mu, [TKF92](crate::tkf_model::tkf92)
    /// has 3 parameters: lambda, mu and r.
    /// The parameter r in [TKF92](crate::tkf_model::tkf92) is used to model the length distribution of inserted segments,
    /// i.e., in [`crate::tkf_model::TKF92IndelModel::insertion_prob_at_non_root`] and
    /// [`super::TKF92IndelModel::insertion_prob_at_root`].
    fn params(&self) -> &[f64];
    fn set_param(&mut self, idx: usize, value: f64);
    fn param_range(&self, idx: usize) -> ParamRange;
    /// Returns the factor corresponding to an insertion event at the root.
    fn insertion_prob_at_root(&self) -> f64;
    /// Returns the factor corresponding to an insertion event at a non-root node.
    fn insertion_prob_at_non_root(&self, beta: f64) -> f64;
    /// Given the subtree event probability for the root (i.e., the tree event probability)
    /// and the block length, returns the log probability of the block under the model.
    fn block_prob(&self, tree_event_prob: f64, block_len: usize) -> f64;
    fn get_blocks<AA: AncestralAlignment>(msa: &AA) -> Vec<usize>;
}

// TODO: link our paper once it is published. For now see original TKF92 paper: https://doi.org/10.1007/bf00163848
/// This struct holds intermediate values for the computation of the log likelihood
/// of an ancestral alignment and tree under a TKF indel model, i.e., without substitutions.
/// The intermediate values are needed for re-alignment.
#[derive(Clone, Debug)]
pub(super) struct TKFIndelModelInfo {
    /// node_event_prob[(node, block)] = the probability factor for the event
    /// on the edge above <node> for the block with id <block>.
    /// See [`TKFIndelCost::event_factor_for_root`] and
    /// [`TKFIndelCost::event_factor_for_non_root`].
    node_event_prob: DMatrix<f64>,
    /// subtree_event_prob[(node, block)] = the product of the event probability factors
    /// for all edges in the subtree rooted in <node> for the block with id <block>,
    /// including the edge above <node>.
    /// See [`TKFIndelCost::set_node_values`].
    subtree_event_prob: DMatrix<f64>,

    /// node_eta[(node, block)] = node_eta[(node, block)] = eta if the current event is an
    /// insertion and the previous one was a deletion, 0 otherwise.
    /// See [`TKFIndelCost::eta_for_non_root`].
    node_eta: DMatrix<f64>,
    /// subtree_eta[(node, block)] = sum of node_eta for all nodes in the subtree rooted in <node>
    /// for the block with id <block>. Since we only have one insertion per column, at most one
    /// node in the subtree can contribute to this sum.
    subtree_eta: DMatrix<f64>,

    /// beta[node] = beta(node.blen)), precomputed for each node.
    /// See [`beta`] function.
    beta: Vec<f64>,
    /// n0[node] = n0(node.blen), precomputed for each node.
    /// See [`n0`] function.
    n0: Vec<f64>,
    /// h1[node] = h1(node.blen), precomputed for each node.
    /// See [`h1`] function.
    h1: Vec<f64>,
    /// insertion[node], precomputed for each node.
    /// See [`TKFModel::insertion_prob_at_root`] and [`TKFModel::insertion_prob_at_non_root`].
    insertion: Vec<f64>,
    /// eta[node] = n1/ (n0 * lambda * beta(node.blen)), precomputed for each node.
    /// See [`eta`] function.
    eta: Vec<f64>,

    /// The right exclusive interval borders of the blocks.
    /// See [`TKFModel::get_blocks`].
    blocks: Vec<usize>,
    /// The lengths of the blocks.
    /// See [`get_block_lengths`].
    block_lengths: Vec<usize>,

    /// previous_event_deletion[node] = true if the last event was a deletion for a that <node>.
    /// See [`TKFIndelCost::determine_event`] and [`TKFIndelCost::update_previous_event`].
    previous_event_deletion: FixedBitSet,

    /// valid[node] = true if the intermediate values for that <node> are valid.
    valid: FixedBitSet,
}

impl TKFIndelModelInfo {
    pub(super) fn new<AA: AncestralAlignment, T: TKFModel>(
        phylo: &PhyloInfo<AA>,
    ) -> TKFIndelModelInfo {
        let blocks = T::get_blocks(&phylo.msa);
        let block_lengths = get_block_lengths(&blocks);
        let n_blocks = blocks.len();
        let n_nodes = phylo.tree.len();
        TKFIndelModelInfo {
            node_event_prob: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            subtree_event_prob: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            node_eta: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            subtree_eta: DMatrix::<f64>::zeros(n_nodes, n_blocks),
            beta: vec![0.0; n_nodes],
            n0: vec![0.0; n_nodes],
            h1: vec![0.0; n_nodes],
            insertion: vec![0.0; n_nodes],
            eta: vec![0.0; n_nodes],
            blocks,
            block_lengths,
            previous_event_deletion: FixedBitSet::with_capacity(n_nodes),
            valid: FixedBitSet::with_capacity(n_nodes),
        }
    }
}

/// Computes the log likelihood of an ancestral alignment and tree under a TKF indel model,
/// i.e., without substitutions. The model is generic over the specific TKF model (e.g., TKF91 or
/// TKF92).
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
    pub(super) fn logl(&self) -> f64 {
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

        let lambda = self.model.lambda();
        let mu = self.model.mu();
        let root_id = usize::from(self.phylo.tree.root);
        let mut logl = 0.0;
        logl += (1.0 - lambda / mu).ln();
        let model_info = self.model_info.borrow();
        for node in self.phylo.tree.postorder() {
            if node == &self.phylo.tree.root {
                continue;
            }
            logl += log_i1(lambda, model_info.beta[usize::from(node)]);
        }
        for block_id in 0..model_info.blocks.len() {
            let block_len = model_info.block_lengths[block_id];
            logl += model_info.subtree_eta[(root_id, block_id)];
            let tree_event_prob = model_info.subtree_event_prob[(root_id, block_id)];
            logl += self.model.block_prob(tree_event_prob, block_len);
        }
        logl
    }

    fn set_root(&self) {
        let root_idx = &self.phylo.tree.root;
        if self.model_info.borrow().valid[usize::from(root_idx)] {
            return;
        }
        self.reset_cache(root_idx);
        let n_blocks = self.model_info.borrow().blocks.len();
        for block_id in 0..n_blocks {
            let x = self.event_prob_for_root(block_id);
            self.set_node_values(root_idx, block_id, x, 0.0);
        }
        self.model_info
            .borrow_mut()
            .valid
            .set(usize::from(root_idx), true);
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
            let node_event_prob = self.event_prob_for_non_root(node_idx, event);
            let node_eta = self.eta_for_non_root(node_idx, event);
            self.set_node_values(node_idx, block_id, node_event_prob, node_eta);
            self.update_previous_event(node_idx, event);
        }

        let mut model_info = self.model_info.borrow_mut();
        if let Some(parent_idx) = self.phylo.tree.parent(node_idx) {
            model_info.valid.set(usize::from(parent_idx), false);
        }
        model_info.valid.set(node_id, true);
    }

    fn update_previous_event(&self, node_idx: &NodeIdx, action: Event) {
        let node_id = usize::from(node_idx);
        let mut model_info = self.model_info.borrow_mut();
        match action {
            Event::Deletion => model_info.previous_event_deletion.set(node_id, true),
            Event::Insertion | Event::Homolog => {
                model_info.previous_event_deletion.set(node_id, false)
            }
            // Since nothing happened, the previous event status remains the same.
            Event::Nothing => {}
        }
    }

    fn reset_cache(&self, node_idx: &NodeIdx) {
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
            self.model.insertion_prob_at_root()
        } else {
            self.model.insertion_prob_at_non_root(beta)
        };
        model_info.previous_event_deletion.set(node_id, false);
        model_info.eta[node_id] = eta(lambda, mu, beta, model_info.n0[node_id], blen);
        model_info.valid.set(node_id, false);
    }

    fn set_node_values(
        &self,
        node_idx: &NodeIdx,
        block_id: usize,
        node_event_prob: f64,
        node_eta: f64,
    ) {
        let node_id = usize::from(node_idx);
        let mut model_info = self.model_info.borrow_mut();
        model_info.node_event_prob[(node_id, block_id)] = node_event_prob;
        model_info.node_eta[(node_id, block_id)] = node_eta;
        let mut substree_event_prob = node_event_prob;
        let mut subtree_eta = node_eta;
        for child in &self.phylo.tree.node(node_idx).children {
            let child_id = usize::from(child);
            substree_event_prob *= model_info.subtree_event_prob[(child_id, block_id)];
            subtree_eta += model_info.subtree_eta[(child_id, block_id)];
        }
        model_info.subtree_event_prob[(node_id, block_id)] = substree_event_prob;
        model_info.subtree_eta[(node_id, block_id)] = subtree_eta;
    }

    fn event_prob_for_root(&self, block_id: usize) -> f64 {
        let root_idx = &self.phylo.tree.root;
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let char_present_at_root = self.phylo.msa.ancestral_map(root_idx)[site].is_some();
        if char_present_at_root {
            return self.model_info.borrow().insertion[usize::from(root_idx)];
        }
        1.0
    }

    /// Determines the event that happened on the edge above `node_idx` for the given `block_id`
    /// based on the ancestral alignment.
    fn determine_event(&self, node_idx: &NodeIdx, block_id: usize) -> Event {
        let parent_idx = self.phylo.tree.node(node_idx).parent.unwrap();
        // the presence or absence of characters is the same for all sites in a block
        // so we can just check the last site of the block
        let site = self.model_info.borrow().blocks[block_id] - 1;
        let parent_is_gap = match parent_idx {
            Internal(_) => self.phylo.msa.ancestral_map(&parent_idx)[site].is_none(),
            _ => unreachable!("The parent of a node cannot be a leaf."),
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

    /// Returns eta if the current event is an insertion and the previous one was a deletion, 0
    /// otherwise.
    /// See [`eta`] function.
    /// Since there can't be a deletion at the root (it has no parent),
    /// this function is only for non-root nodes.
    fn eta_for_non_root(&self, node_idx: &NodeIdx, event: Event) -> f64 {
        if matches!(event, Event::Insertion)
            && self.model_info.borrow().previous_event_deletion[usize::from(node_idx)]
        {
            self.model_info.borrow().eta[usize::from(node_idx)]
        } else {
            0.0
        }
    }

    fn event_prob_for_non_root(&self, node_idx: &NodeIdx, action: Event) -> f64 {
        let node_id = usize::from(node_idx);
        match action {
            Event::Deletion => self.model_info.borrow().n0[node_id],
            Event::Homolog => self.model_info.borrow().h1[node_id],
            Event::Insertion => self.model_info.borrow().insertion[node_id],
            Event::Nothing => 1.0,
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
        // TODO: At the time of writing this, this method is only used to set the frequencies of
        // the model, but the TKF92IndelCost does not have frequencies.
        self.phylo.freqs()
    }

    fn freqs(&self) -> &FreqVector {
        &DUMMY_FREQS
    }
}

/// Returns the value of beta(t) for a branch of length `time`.
/// It is called beta(t) in the TKF papers.
#[inline]
pub(super) fn beta(lambda: f64, mu: f64, time: f64) -> f64 {
    let exp_term = ((lambda - mu) * time).exp();
    (1.0 - exp_term) / (mu - lambda * exp_term)
}

/// Returns the log probability of a character being inserted right of the immortal link
/// along a branch of length `time`, i.e., at the very left of the sequence.
/// The 'time' is implicitly included in beta.
/// It is called p''_1() in the TKF papers.
#[inline]
pub(super) fn log_i1(lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta).ln()
}

/// Returns the probability of a homologous character surviving along a branch of length `time`.
/// The 'time' is also implicitly included in beta.
/// It is called p_1(t) in the TKF papers.
#[inline]
pub(super) fn h1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    (-mu * time).exp() * (1.0 - lambda * beta)
}

/// Returns the probability of a character being deleted along a branch of length `time`.
/// It is called p'_0(t) in the TKF papers.
/// The 'time' is also implicitly included in beta.
#[inline]
pub(super) fn n0(mu: f64, beta: f64) -> f64 {
    mu * beta
}

/// Returns the log probability of a new character being inserted right of a character that is
/// deleted along a branch of length `time`.
/// The 'time' is also implicitly included in beta.
/// It is called p'_1() in the TKF papers.
#[inline]
pub(super) fn log_n1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    ((1.0 - (-mu * time).exp() - mu * beta) * (1.0 - lambda * beta)).ln()
}

/// Returns the log of the n1 / (n0 * lambda * beta).
/// This is used in the case where an insertion follows a deletion,
/// since the event factors included n0 for the deletion and lambda * beta for the insertion
/// but under the TKF model they are not independent and instead n1 should be used.
/// Eta corrects for that.
/// The 'time' is also implicitly included in beta and n0.
#[inline]
pub(super) fn eta(lambda: f64, mu: f64, beta: f64, n0: f64, time: f64) -> f64 {
    let mut eta = log_n1(lambda, mu, beta, time);
    eta -= (lambda * beta).ln();
    eta -= n0.ln();
    eta
}

/// Given the right exclusive block borders, returns the lengths of the blocks.
/// For example, given [3, 5, 8], the block lengths are [3, 2, 3].
pub(super) fn get_block_lengths(blocks: &[usize]) -> Vec<usize> {
    let mut block_lens = vec![0; blocks.len()];
    for (i, block) in blocks.iter().enumerate() {
        block_lens[i] = if i == 0 {
            *block
        } else {
            block - blocks[i - 1]
        };
    }
    block_lens
}

#[cfg(test)]
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
