use std::cell::RefCell;
use std::collections::VecDeque;

use hashbrown::HashMap;
use log::warn;
use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Geometric};

use crate::alignment::{Alignment, AlignmentSimulation, AncestralAlignment, Sequences, MASA};
use crate::alphabets::{AMB_CHAR, GAP};
use crate::random::RandomGenerator;
use crate::tkf_model::simulate_msa::Fragmentation;
use crate::tkf_model::{beta, h1, n0, TKFModel};
use crate::tree::{NodeIdx, Tree};
use crate::{record_wo_desc as record, Result};

/// Abstracts over how a single fragment's length is sampled.
///
/// [TKF92](`crate::tkf_model::TKF92IndelModel`) draws from a geometric distribution parameterised by `r`.
/// [TKF91](`crate::tkf_model::TKF91IndelModel`) always returns length 1 (each residue is its own independent link).
///
/// Returns `(length, log_probability)` so that the caller can accumulate
/// the simulation log-likelihood without any model-specific logic.
pub trait FragmentSampler {
    fn sample_fragment_length<R: Rng>(&self, rng: &mut R) -> (usize, f64);
}

/// Computes the expected length of the root sequence (number of characters)
/// under the TKF model if the process has reached stationarity.
pub trait ExpectedRootLength {
    fn expected_root_length(&self) -> f64;
}

/// Since these sequences are built incrementally, we can't use the [`Sequences`] which hold immutable [records](`record`).
type Seqs = HashMap<NodeIdx, Vec<u8>>;

#[derive(Clone, Copy, Debug)]
pub enum RootLength {
    Sampled,
    Defined(usize),
    Expected,
}

/// Simulates the indel process under a [TKFModel](crate::tkf_model::TKFModel) to produce an MASA
/// (= multiple ancestral sequence alignment) containing only [`GAP`]s and [`AMB_CHAR`]s, representing the indel history.
/// The `max_insertion_length` parameter controls the maximum number of inserted links (=fragments)
/// that can be produced in a single insertion event.
/// Note that, the MASA might contain columns where the character goes extinct, i.e., all leaf
/// sequences have a gap in that column. You may want to call
/// [`AncestralAlignment::remove_extinct_columns`](`crate::alignment::AncestralAlignment::remove_extinct_columns`) on the
/// resulting alignment or
/// [`TKFIndelMSASimulationResult::remove_extinct_columns`](`TKFIndelMSASimulationResult::remove_extinct_columns`) on the
/// simulation result if you also care about the fragmentation and want to remove those.
pub struct TKFIndelMSASimulator<
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore,
> {
    indel_model: T,
    tree: Tree,
    cumulative_logl: RefCell<f64>,
    rng: RefCell<RandomGenerator<R>>,
    max_insertion_length: usize,
    root_length: RootLength,
}

impl<T, R> AlignmentSimulation for TKFIndelMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(&self) -> AA {
        let TKFIndelMSASimulationResult { masa: msa, .. } = self.simulate_with_fragments();
        msa
    }

    fn simulate_alignment<A: Alignment>(&self) -> A {
        self.simulate_ancestral_alignment::<MASA>()
            .into_alignment(&self.tree)
    }
}

/// All the descendant links associated to a single branch.
type BranchLinkChildren = Vec<TKFLink>;

struct TKFLink {
    /// The node in the tree the link is associated with.
    node: NodeIdx,
    /// True if the link is immortal. It's the link to the very left of the sequence.
    is_immortal: bool,
    /// How many characters are associated to the right of this link.
    length: usize,
    /// For every child node/branch of this link's node, the descendant links.
    children: Vec<BranchLinkChildren>,
    /// For every child node/branch of this link's node, the [`LinkFate`] of this link.
    fates: Vec<LinkFate>,
    /// Whether this link was produced by an insertion event.
    is_insertion: bool,
}

impl TKFLink {
    fn new_immortal(node: NodeIdx) -> Self {
        TKFLink {
            node,
            is_immortal: true,
            length: 0,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: false,
        }
    }
    fn new(node: NodeIdx, length: usize) -> Self {
        TKFLink {
            node,
            is_immortal: false,
            length,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: false,
        }
    }

    fn new_insertion(node: NodeIdx, length: usize) -> Self {
        TKFLink {
            node,
            is_immortal: false,
            length,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: true,
        }
    }
}

#[derive(Clone)]
enum LinkFate {
    /// The link is deleted.
    Deletion,
    /// The link survives on the branch and produces `usize` many insertions to the right of it.
    Homolog(usize),
    /// The link is deleted but produces `usize` many insertions to the right of it.
    NonHomolog(usize),
}

pub struct TKFIndelMSASimulationResult<AA: AncestralAlignment> {
    pub masa: AA,
    pub fragmentation: Fragmentation,
}

impl<AA: AncestralAlignment> TKFIndelMSASimulationResult<AA> {
    pub fn remove_extinct_columns(&mut self) -> Result<()> {
        let keep_col_mask = self.masa.remove_extinct_columns();
        self.fragmentation.remove_cols(&keep_col_mask)
    }
}

impl<T, R> TKFIndelMSASimulator<T, R>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng + RngCore,
{
    pub fn new(
        indel_model: T,
        tree: Tree,
        rng: RandomGenerator<R>,
        max_insertion_length: usize,
    ) -> Self {
        Self {
            indel_model,
            tree,
            cumulative_logl: RefCell::new(0.0),
            rng: RefCell::new(rng),
            max_insertion_length,
            root_length: RootLength::Sampled,
        }
    }

    /// Sets a defined root length for the simulation. If `None`, the root length is sampled.
    pub fn root_length(&mut self, root_length: RootLength) -> &mut Self {
        self.root_length = root_length;
        self
    }

    pub(super) fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Simulates the indel process and returns the alignment, fragmentation, and log-likelihood of the simulated MSA
    /// under the model, which is accumulated during the simulation.
    fn simulate_with_fragments_and_logl<AA: AncestralAlignment>(
        &self,
    ) -> (TKFIndelMSASimulationResult<AA>, f64) {
        *self.cumulative_logl.borrow_mut() = 0.0;
        let links = self.build_msa_links();
        let (masa, fragmentation) = self.links_to_msa(&links);
        let logl = *self.cumulative_logl.borrow();
        let result = TKFIndelMSASimulationResult {
            masa,
            fragmentation,
        };
        (result, logl)
    }

    /// Simulates the indel process and produces the corresponding ancestral alignment and fragmentation.
    pub fn simulate_with_fragments<AA: AncestralAlignment>(
        &self,
    ) -> TKFIndelMSASimulationResult<AA> {
        self.simulate_with_fragments_and_logl().0
    }

    /// Samples the number of root links from a geometric distribution. Can be zero.
    fn sample_num_root_links(&self) -> usize {
        let prob_of_success = 1.0 - self.indel_model.lambda() / self.indel_model.mu(); // ie stopping the links
        let geom = Geometric::new(prob_of_success).unwrap();
        let choice = geom.sample(&mut self.rng.borrow_mut().rng);
        let prob = (1.0 - prob_of_success).powi(choice as i32) * prob_of_success;
        *self.cumulative_logl.borrow_mut() += prob.ln();
        choice as usize
    }

    /// Samples the length of a fragment from the indel model's fragment length distribution, and
    /// accumulates the log-probability of that length into the cumulative log-likelihood of the simulation.
    fn sample_fragment_length(&self) -> usize {
        let (length, log_prob) = self
            .indel_model
            .sample_fragment_length(&mut self.rng.borrow_mut().rng);
        *self.cumulative_logl.borrow_mut() += log_prob;
        length
    }

    /// Builds the links associated with the root of the tree. This includes one immortal
    /// link and a number of mortal links either sampled or defined.
    fn build_root_links(&self) -> Vec<TKFLink> {
        let min_needed_links = match self.root_length {
            RootLength::Sampled => self.sample_num_root_links(),
            RootLength::Defined(len) => len,
            RootLength::Expected => self.indel_model.expected_root_length().round() as usize,
        };
        let max_root_seq_len = match self.root_length {
            RootLength::Defined(len) => Some(len),
            RootLength::Expected => Some(self.indel_model.expected_root_length().round() as usize),
            RootLength::Sampled => None,
        };
        let mut root_links = Vec::with_capacity(min_needed_links + 1); // +1 for the immortal link
        root_links.push(TKFLink::new_immortal(self.tree.root));

        let mut current_total_root_len = 0;
        for _ in 0..min_needed_links {
            let frag_length = self.sample_fragment_length();
            match max_root_seq_len {
                Some(max_len) => {
                    if current_total_root_len + frag_length <= max_len {
                        root_links.push(TKFLink::new(self.tree.root, frag_length));
                        current_total_root_len += frag_length;
                    } else {
                        let remaining_len = max_len - current_total_root_len;
                        root_links.push(TKFLink::new(self.tree.root, remaining_len));
                        break;
                    }
                }
                _ => {
                    root_links.push(TKFLink::new(self.tree.root, frag_length));
                }
            }
        }
        root_links
    }

    /// First [builds the root links](`Self::build_root_links`)) and then
    /// [evolves them down the tree](`Self::evolve_link_down_tree`) to produce the full link
    /// structure of the MSA.
    fn build_msa_links(&self) -> Vec<TKFLink> {
        let mut root_links = self.build_root_links();
        for link in root_links.iter_mut() {
            // TODO: this can be parallelized. Also, the conversion to to alignment, i.e., the
            // progressive appending to it can be done after a root link is done evolving down the
            // tree, which would allow for more efficient memory usage since we won't have to keep
            // the whole link structure in memory at once.
            // See issue #163 https://github.com/acg-team/rust-phylo/issues/163
            self.evolve_link_down_tree(link);
        }
        root_links
    }

    /// Evolves a single link down the tree, by sampling its fate on every branch and creating the
    /// descendant links accordingly.
    fn evolve_link_down_tree(&self, link: &mut TKFLink) {
        for (branch_id, child_node) in self.tree.children(&link.node).iter().enumerate() {
            link.children.push(Vec::new());
            let branch_length = self.tree.node(child_node).blen;
            if link.is_immortal {
                //  immortal link always survives and evolves down the tree
                let child_link = TKFLink::new_immortal(*child_node);
                link.children[branch_id].push(child_link);
                self.evolve_link_down_tree(link.children[branch_id].last_mut().unwrap());
                // new insertions
                let num_insertions = self.sample_tkf_immortal_link_fate(branch_length);
                self.evolve_insertions(num_insertions, link, branch_id);
            } else {
                let fate = self.sample_tkf_link_fate(branch_length);
                link.fates.push(fate.clone());
                match fate {
                    LinkFate::Deletion => {
                        // link is deleted, do nothing
                    }
                    LinkFate::Homolog(num_children) => {
                        // the surviving homologous link
                        let child_link = TKFLink::new(*child_node, link.length);
                        link.children[branch_id].push(child_link);
                        self.evolve_link_down_tree(link.children[branch_id].last_mut().unwrap());
                        // new insertions
                        self.evolve_insertions(num_children, link, branch_id);
                    }
                    LinkFate::NonHomolog(num_children) => {
                        // original link is deleted, do not evolve_link_down_tree
                        // new insertions
                        self.evolve_insertions(num_children, link, branch_id);
                    }
                }
            }
        }
    }

    /// A helper for [evolve_link_down_tree](`Self::evolve_link_down_tree`) that evolves a `number`
    /// of insertions on a branch, by creating the insertion links and evolving them down the tree
    /// as well.
    fn evolve_insertions(&self, number: usize, parent_link: &mut TKFLink, branch_id: usize) {
        for _ in 0..number {
            let length = self.sample_fragment_length();
            let node = self.tree.children(&parent_link.node)[branch_id];
            let insertion_link = TKFLink::new_insertion(node, length);
            parent_link.children[branch_id].push(insertion_link);
            self.evolve_link_down_tree(parent_link.children[branch_id].last_mut().unwrap());
        }
    }

    /// Samples the fate of a mortal link on a branch, which includes whether it survives or is
    /// deleted, and how many insertions it produces in either case.
    fn sample_tkf_link_fate(&self, time: f64) -> LinkFate {
        let uniform_sample = self.rng.borrow_mut().random::<f64>();
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = beta(lambda, mu, time);
        let n_0 = n0(self.indel_model.mu(), beta);
        let homolog_prob_integrated = homolog_prob_integrated(mu, time);
        let non_homolog_prob_integrated = non_homolog_prob_integrated(mu, beta, time);
        debug_assert!(
            (n_0 + homolog_prob_integrated + non_homolog_prob_integrated - 1.0).abs() < 1e-10
        );
        // Link is deleted without producing any insertions
        if uniform_sample < n_0 {
            *self.cumulative_logl.borrow_mut() += (n_0).ln();
            LinkFate::Deletion
        } else if uniform_sample < n_0 + homolog_prob_integrated {
            // Link survives homologously
            let adjusted_sample = uniform_sample - n_0;
            self.sample_homolog_fate(time, adjusted_sample)
        } else {
            // Link is deleted but has non-homologous insertions
            debug_assert!(1.0 - uniform_sample < non_homolog_prob_integrated);
            let adjusted_sample = uniform_sample - n_0 - homolog_prob_integrated;
            self.sample_non_homolog_fate(time, adjusted_sample)
        }
    }

    /// Samples the number of homologous insertions on a branch, given that the original link survives.
    fn sample_homolog_fate(&self, time: f64, uniform_sample: f64) -> LinkFate {
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = beta(lambda, mu, time);
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_insertion_length + 1 {
            let homolog_prob = homolog_prob(n, lambda, mu, beta, time);
            cumulative_prob += homolog_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += homolog_prob.ln();
                return LinkFate::Homolog(n - 1); // n - 1 because we are not counting the original link
            }
        }
        // didn't return in the loop, capping insertion at length max_insertion_length
        let homolog_prob_integrated = homolog_prob_integrated(mu, time);
        let homolog_prob = homolog_prob_integrated - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += homolog_prob.ln();
        warn!(
            "Capping homologous insertion length at {}",
            self.max_insertion_length
        );
        LinkFate::Homolog(self.max_insertion_length)
    }

    /// Samples the number of non-homologous insertions on a branch, given that the original link is deleted.
    fn sample_non_homolog_fate(&self, time: f64, uniform_sample: f64) -> LinkFate {
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = beta(lambda, mu, time);
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_insertion_length {
            let non_homolog_prob = non_homolog_prob(n, lambda, mu, beta, time);
            cumulative_prob += non_homolog_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += non_homolog_prob.ln();
                return LinkFate::NonHomolog(n);
            }
        }
        // didn't return in the loop, capping insertion at length max_insertion_length
        let non_homolog_prob_integrated = non_homolog_prob_integrated(mu, beta, time);
        let non_homolog_prob = non_homolog_prob_integrated - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += non_homolog_prob.ln();
        warn!(
            "Capping non-homologous insertion length at {}",
            self.max_insertion_length
        );
        LinkFate::NonHomolog(self.max_insertion_length)
    }

    /// Since immortal links always survive, we only need to sample how many insertions they
    /// produce, which is what this function does.
    fn sample_tkf_immortal_link_fate(&self, time: f64) -> usize {
        let uniform_sample = self.rng.borrow_mut().random::<f64>();
        let lambda = self.indel_model.lambda();
        let beta = beta(lambda, self.indel_model.mu(), time);
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_insertion_length {
            let immortal_prob = immortal_prob(n, lambda, beta);
            cumulative_prob += immortal_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
                return n - 1;
            }
        }
        let immortal_prob = 1.0 - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
        self.max_insertion_length
    }

    /// After [simulation of the links](`Self::build_msa_links`) is complete, this function
    /// calls [`Self::append_link_to_msa`] for every root link to build the final MSA and fragmentation.
    fn links_to_msa<AA: AncestralAlignment>(&self, links: &Vec<TKFLink>) -> (AA, Fragmentation) {
        let mut msa: Seqs = HashMap::new();
        for node in self.tree.preorder() {
            msa.insert(*node, Vec::new());
        }
        let mut fragmentation: Vec<usize> = Vec::new();
        for link in links {
            self.append_link_to_msa(link, &mut msa, &mut fragmentation);
        }

        let records = msa
            .iter()
            .map(|(node, seq)| record!(&self.tree.node(node).id, seq))
            .collect();

        let seqs = Sequences::new(records);
        let msa = AA::from_aligned_with_ancestral(seqs, &self.tree).unwrap();

        let fragmentation = Fragmentation::new(fragmentation)
            .expect("fragmentation should be valid since it was built from the links, pls report this at...");
        fragmentation.fragmentation_works_with_ancestral_alignment(&msa).expect("fragmentation should work with ancestral alignment since it was built from the links, pls report this at...");
        (msa, fragmentation)
    }

    /// Traverses the link structure on the tree for every root link and progressively builds the alignment.
    fn append_link_to_msa(&self, link: &TKFLink, msa: &mut Seqs, fragmentation: &mut Vec<usize>) {
        // Insertions are pushed to the front of the queue so that they the lower insertions on the
        // tree are processed before the higher ones, which ensures that gaps are correctly
        // inserted since lower insertions don't mess with the events higher up in the tree.
        // Which is especially important for the case of non-homologous insertions and the use of eta.
        let mut insertions = VecDeque::from([link]);
        while let Some(insertion_link) = insertions.pop_front() {
            // For every insertion event do a BFS traversal, preorder would not work since we need
            // to maintain the priorities of the insertion events, see the comment above.
            let mut tree_stack = VecDeque::from([insertion_link]);
            while let Some(current_link) = tree_stack.pop_front() {
                if current_link.is_immortal {
                    self.dispatch_immortal_children(current_link, &mut tree_stack, &mut insertions);
                } else {
                    self.process_link(
                        current_link,
                        msa,
                        fragmentation,
                        &mut tree_stack,
                        &mut insertions,
                    );
                }
            }
        }
    }

    /// Pushes the children of an immortal link onto the traversal stacks.
    ///
    /// The first child of each branch continues the homolog tree traversal (`tree_stack`) which is
    /// the immortal surviving; any additional children are new insertions and go onto the `insertions` queue.
    fn dispatch_immortal_children<'a>(
        &self,
        link: &'a TKFLink,
        tree_stack: &mut VecDeque<&'a TKFLink>,
        insertions: &mut VecDeque<&'a TKFLink>,
    ) {
        debug_assert!(link.is_immortal);
        for branch_child in &link.children {
            for (child_id, child_link) in branch_child.iter().enumerate() {
                if child_id == 0 {
                    tree_stack.push_back(child_link);
                } else {
                    insertions.push_front(child_link);
                }
            }
        }
    }

    /// Writes one non-immortal link's contribution to the MSA and updates the traversal stacks.
    ///
    /// - Appends `AMB_CHAR * length` to the owning node.
    /// - Records a fragmentation boundary for root-owned links and insertion links.
    /// - Fills `NOTHING_CHAR` gaps into all non-owning nodes for insertion links.
    /// - Applies each branch fate: queues homolog/insertion children, or fills deletion gaps.
    fn process_link<'a>(
        &self,
        link: &'a TKFLink,
        msa: &mut Seqs,
        fragmentation: &mut Vec<usize>,
        tree_stack: &mut VecDeque<&'a TKFLink>,
        insertions: &mut VecDeque<&'a TKFLink>,
    ) {
        debug_assert!(!link.is_immortal);
        let seq = msa.get_mut(&link.node).unwrap();
        let new_len = seq.len() + link.length;
        seq.resize(new_len, AMB_CHAR);
        if link.node == self.tree.root || link.is_insertion {
            let fragment_boundary = link.length + fragmentation.last().unwrap_or(&0);
            fragmentation.push(fragment_boundary);
        }
        if link.is_insertion {
            self.insertion_gaps(link.length, msa, &link.node);
        }
        for (branch_id, child_links) in link.children.iter().enumerate() {
            self.apply_fate(link, branch_id, child_links, msa, tree_stack, insertions);
        }
    }

    /// Applies a single branch fate for a non-immortal link.
    ///
    /// - `Homolog`: queues the surviving child onto `tree_stack` and its insertions onto `insertions`.
    /// - `Deletion` / `NonHomolog`: fills deletion gaps for all descendants; `NonHomolog` also
    ///   queues the new insertion children.
    fn apply_fate<'a>(
        &self,
        link: &'a TKFLink,
        branch_id: usize,
        child_links: &'a [TKFLink],
        msa: &mut Seqs,
        tree_stack: &mut VecDeque<&'a TKFLink>,
        insertions: &mut VecDeque<&'a TKFLink>,
    ) {
        debug_assert!(!link.is_immortal);
        match &link.fates[branch_id] {
            LinkFate::Homolog(_) => {
                tree_stack.push_back(&child_links[0]);
                for l in child_links.iter().skip(1) {
                    insertions.push_front(l);
                }
            }
            LinkFate::Deletion => {
                let child_node = self.tree.children(&link.node)[branch_id];
                for descendant_node in self.tree.preorder_subroot(&child_node) {
                    let seq = msa.get_mut(&descendant_node).unwrap();
                    let new_len = seq.len() + link.length;
                    seq.resize(new_len, GAP);
                }
            }
            LinkFate::NonHomolog(_) => {
                for l in child_links {
                    insertions.push_front(l);
                }
                let child_node = self.tree.children(&link.node)[branch_id];
                for descendant_node in self.tree.preorder_subroot(&child_node) {
                    let seq = msa.get_mut(&descendant_node).unwrap();
                    let new_len = seq.len() + link.length;
                    seq.resize(new_len, GAP);
                }
            }
        }
    }

    /// Appends gaps to the progressively build alignment for every node in the tree that is not in
    /// the subtree of the insertion node.
    fn insertion_gaps(&self, length: usize, msa: &mut Seqs, insertion_node: &NodeIdx) {
        self.insertion_gaps_subtree(length, msa, &self.tree.root, insertion_node);
    }

    /// A helper for [insertion_gaps](`Self::insertion_gaps`) that recursively traverses the tree
    /// and appends gaps to the appropriate nodes.
    fn insertion_gaps_subtree(
        &self,
        length: usize,
        msa: &mut Seqs,
        subtree_node: &NodeIdx,
        insertion_node: &NodeIdx,
    ) {
        if subtree_node == insertion_node {
            return;
        }
        let seq = msa.get_mut(subtree_node).unwrap();
        let new_len = seq.len() + length;
        seq.resize(new_len, GAP);
        for child_node in self.tree.children(subtree_node) {
            self.insertion_gaps_subtree(length, msa, child_node, insertion_node);
        }
    }
}

/// Returns the sum [`homolog_prob`] over all `n` (number of insertions) which is the probability
/// that a link survives on a branch of length `time` regardless of how many insertions it produces.   
fn homolog_prob_integrated(mu: f64, time: f64) -> f64 {
    (-mu * time).exp()
}

/// Returns the sum of [`non_homolog_prob`] over all `n` (number of insertions) which is the
/// probability that a link is deleted on a branch of length `time` but produces at least one insertion.
fn non_homolog_prob_integrated(mu: f64, beta: f64, time: f64) -> f64 {
    1.0 - (-mu * time).exp() - mu * beta
}

/// In the TKF model `n` is at least 1 (the original link)
fn homolog_prob(n: usize, lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let h1_val = h1(lambda, mu, beta, time);
    h1_val * (lambda * beta).powi((n - 1) as i32)
}

/// In the TKF model `n` is at least 1 because otherwise we should have used [`crate::tkf_model::n0`].
fn non_homolog_prob(n: usize, lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let t1 = 1.0 - (-mu * time).exp() - mu * beta;
    let t2 = 1.0 - lambda * beta;
    let t3 = (lambda * beta).powi((n - 1) as i32);
    t1 * t2 * t3
}

/// In the TKF model `n` is at least 1 because the immortal link cannot die
fn immortal_prob(n: usize, lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta) * (lambda * beta).powi((n - 1) as i32)
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::alignment::{Alignment, AncestralAlignment, MASA};
    use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
    use crate::random::DefaultGenerator;
    use crate::tkf_model::{
        beta, n0, TKF91IndelCostBuilder, TKF91IndelModel, TKF92FixedIndelCostBuilder,
        TKF92IndelModel,
    };
    use crate::tree;

    use super::*;

    #[rstest]
    #[case(0.1, 0.2, 0.5)]
    #[case(0.5, 0.7, 1.0)]
    #[case(1.0, 1.5, 0.0001)]
    fn tkf_integrated_probs(#[case] lambda: f64, #[case] mu: f64, #[case] time: f64) {
        // arrange
        let beta = beta(lambda, mu, time);
        //act
        let homolog_integrated = homolog_prob_integrated(mu, time);
        let non_homolog_integrated = non_homolog_prob_integrated(mu, beta, time);
        let n0 = n0(mu, beta);
        // assert
        assert_relative_eq!(
            homolog_integrated + non_homolog_integrated + n0,
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn tkf_homolog_probs() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.23;
        let beta = beta(lambda, mu, time);
        let n = 4;

        let prob = homolog_prob(n, lambda, mu, beta, time);
        let true_prob =
            (-mu * time).exp() * (1.0 - lambda * beta) * (lambda * beta).powi((n - 1) as i32);
        assert_relative_eq!(prob, true_prob);
    }

    #[test]
    fn tkf_non_homolog_probs() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.23;
        let beta = beta(lambda, mu, time);
        let n = 4;

        let prob = non_homolog_prob(n, lambda, mu, beta, time);
        let true_prob = (1.0 - (-mu * time).exp() - mu * beta)
            * (1.0 - lambda * beta)
            * (lambda * beta).powi((n - 1) as i32);
        assert_relative_eq!(prob, true_prob);
    }

    #[test]
    fn tkf_homolog_probs_sum_to_integrated() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.0;
        let beta = beta(lambda, mu, time);
        let homolog_integrated = homolog_prob_integrated(mu, time);

        let mut homolog_sum = 0.0;
        for n in 1..100 {
            homolog_sum += homolog_prob(n, lambda, mu, beta, time);
        }

        assert_relative_eq!(homolog_sum, homolog_integrated, epsilon = 1e-10);
    }

    #[test]
    fn tkf_non_homolog_probs_sum_to_integrated() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.0;
        let beta = beta(lambda, mu, time);
        let non_homolog_integrated = non_homolog_prob_integrated(mu, beta, time);

        let mut non_homolog_sum = 0.0;
        for n in 1..100 {
            non_homolog_sum += non_homolog_prob(n, lambda, mu, beta, time);
        }

        assert_relative_eq!(non_homolog_sum, non_homolog_integrated, epsilon = 1e-10);
    }

    #[test]
    fn tkf92_indel_simulate() {
        for seed in 0..100 {
            let lambda = 1.1;
            let mu = 1.2;
            let r = 0.6;
            let tkf_model = TKF92IndelModel::new(lambda, mu, r);
            let tree = tree!("(((9: 0.220067, (10: 0.150408, 6: 0.150408): 0.069660): 0.054476, (5: 0.033214, 8: 0.033214): 0.241329): 0.725457, ((((3: 0.033362, 1: 0.033362): 0.000893, 7: 0.034255): 0.379610, 4: 0.413865): 0.310280, 2: 0.724145): 0.275855);");
            let tree = set_missing_tree_node_ids(&tree).unwrap();

            // Should be so large that it doesn't kick in. Because otherwise the
            // accumulated logl would not match the clean cost. Since event probabilities
            // are summed together for the case of drawing a max_insertion_length.
            let max_insertion_length = 100;
            let simulator = TKFIndelMSASimulator::new(
                tkf_model,
                tree.clone(),
                DefaultGenerator::new(seed),
                max_insertion_length,
            );
            let (result, logl): (TKFIndelMSASimulationResult<MASA>, f64) =
                simulator.simulate_with_fragments_and_logl();
            let alignment = result.masa;
            assert_eq!(alignment.seq_count() + alignment.ancestral_seqs().len(), 19);
            let phylo = PhyloInfo {
                msa: alignment.clone(),
                tree,
            };
            assert!(
                phylo.check_dollos_constraint().is_ok(),
                "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
            );
            let cost = TKF92FixedIndelCostBuilder::new(
                lambda,
                mu,
                r,
                result.fragmentation.right_exclusive_boundaries().to_vec(),
                phylo,
            )
            .build()
            .unwrap()
            .logl();
            assert_relative_eq!(logl, cost, epsilon = 1e-10);
        }
    }

    #[test]
    fn tkf91_indel_simulate() {
        for seed in 0..100 {
            let tkf_model = TKF91IndelModel::default();
            let lambda = tkf_model.lambda();
            let mu = tkf_model.mu();
            let tree = tree!("(((9: 0.220067, (10: 0.150408, 6: 0.150408): 0.069660): 0.054476, (5: 0.033214, 8: 0.033214): 0.241329): 0.725457, ((((3: 0.033362, 1: 0.033362): 0.000893, 7: 0.034255): 0.379610, 4: 0.413865): 0.310280, 2: 0.724145): 0.275855);");
            let tree = set_missing_tree_node_ids(&tree).unwrap();

            // Should be so large that it doesn't kick in. Because otherwise the
            // accumulated logl would not match the clean cost. Since event probabilities
            // are summed together for the case of drawing a max_insertion_length.
            let max_insertion_length = 100;
            let simulator = TKFIndelMSASimulator::new(
                tkf_model,
                tree.clone(),
                DefaultGenerator::new(seed),
                max_insertion_length,
            );
            let (result, logl): (TKFIndelMSASimulationResult<MASA>, f64) =
                simulator.simulate_with_fragments_and_logl();
            let alignment = result.masa;
            assert_eq!(alignment.seq_count() + alignment.ancestral_seqs().len(), 19);

            // In TKF91 every residue is its own independent link (fragment length == 1), so the
            // fragmentation must be exactly [1, 2, 3, ..., n_cols].
            let n_cols = result.fragmentation.len();
            let expected_fragmentation: Vec<usize> = (1..=n_cols).collect();
            assert_eq!(
                result.fragmentation.right_exclusive_boundaries(),
                expected_fragmentation,
                "TKF91 fragmentation must have every column as its own block"
            );

            // The simulation logl must equal the TKF91 indel cost for the same alignment.
            let phylo = PhyloInfo {
                msa: alignment.clone(),
                tree,
            };
            assert!(
                phylo.check_dollos_constraint().is_ok(),
                "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
            );
            let cost = TKF91IndelCostBuilder::new(lambda, mu, phylo)
                .build()
                .unwrap()
                .logl();
            assert_relative_eq!(logl, cost, epsilon = 1e-10);
        }
    }

    #[test]
    fn tkf92_indel_sim_consistency() {
        let lambda = 0.29;
        let mu = 0.3;
        let r = 0.9;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        let max_len = 2;
        let seed = 123;
        let simulator1 = TKFIndelMSASimulator::new(
            tkf_model.clone(),
            tree.clone(),
            DefaultGenerator::new(seed),
            max_len,
        );
        let simulator2 =
            TKFIndelMSASimulator::new(tkf_model, tree, DefaultGenerator::new(seed), max_len);
        let result1 = simulator1.simulate_with_fragments::<MASA>();
        let msa2: MASA = simulator2.simulate_ancestral_alignment();
        assert_eq!(result1.masa.to_string(), msa2.to_string());
    }

    #[test]
    fn tkf92_indel_capping() {
        let lambda = 0.19;
        let mu = 0.2;
        let r = 0.5;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        let max_len = 0; // Cap at 0 insertions on branches
        let simulator =
            TKFIndelMSASimulator::new(tkf_model, tree.clone(), DefaultGenerator::new(123), max_len);
        let result = simulator.simulate_with_fragments::<MASA>();
        let msa = result.masa;
        // With max_len = 0, no insertions can happen on branches.
        // Thus, every column in the MSA must have a char at the root.
        let root_map = msa.ancestral_map(&tree.root);
        for (col_idx, site) in root_map.iter().enumerate() {
            assert!(
                site.is_some(),
                "Column {} should have a character at the root",
                col_idx
            );
        }
    }
}
