use std::fmt::Display;
use std::num::NonZeroUsize;

use itertools::Itertools;
use log::{debug, info};

use crate::alignment::Alignment;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{
    BranchOptimiser, MoveCostInfo, MoveOptimiser, NniOptimiser, PhyloOptimisationResult,
    SprOptimiser,
};
use crate::parsimony::scoring::ParsimonyScoring;
use crate::parsimony::{BasicParsimonyCost, DolloParsimonyCost};
use crate::pip_model::PIPCost;
use crate::random::RandomSource;
use crate::substitution_models::{QMatrix, SubstitutionCost};
use crate::tree::NodeIdx;
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub enum TopologyOptimiserPredicate {
    GtEpsilon(f64),
    FixedIter(NonZeroUsize),
    // NOTE: use of `fn(..) -> ..` disallows closures that capture any
    // surrounding variables, for that we would need to allow Boxed Fn
    // trait objects (or introduce a generic parameter which might get tedious)
    Custom(fn(usize, f64) -> bool),
}

impl TopologyOptimiserPredicate {
    fn test(&self, iteration: usize, delta: f64) -> bool {
        use TopologyOptimiserPredicate::*;
        match *self {
            GtEpsilon(min_delta) => delta > min_delta,
            FixedIter(max) => max.get() > iteration,
            Custom(pred) => pred(iteration, delta),
        }
    }
    pub fn gt_epsilon(epsilon: f64) -> Self {
        Self::GtEpsilon(epsilon)
    }
    pub fn fixed_iter(num: NonZeroUsize) -> Self {
        Self::FixedIter(num)
    }
    pub fn custom(pred: fn(usize, f64) -> bool) -> Self {
        Self::Custom(pred)
    }
}

/// The `Compatible` trait is used to ensure that the cost and move optimiser passed to
/// [`TopologyOptimiser::new`] are compatible.
pub trait Compatible<MO: MoveOptimiser> {}

// TODO: or to we want to place those in respective files?
impl<Q: QMatrix, A: Alignment> Compatible<SprOptimiser> for PIPCost<Q, A> {}
impl<Q: QMatrix, A: Alignment> Compatible<NniOptimiser> for PIPCost<Q, A> {}
impl<Q: QMatrix, A: Alignment> Compatible<SprOptimiser> for SubstitutionCost<Q, A> {}
impl<Q: QMatrix, A: Alignment> Compatible<NniOptimiser> for SubstitutionCost<Q, A> {}
impl<S: ParsimonyScoring, A: Alignment> Compatible<SprOptimiser> for DolloParsimonyCost<S, A> {}
impl<S: ParsimonyScoring, A: Alignment> Compatible<NniOptimiser> for DolloParsimonyCost<S, A> {}
impl<A: Alignment> Compatible<SprOptimiser> for BasicParsimonyCost<A> {}
impl<A: Alignment> Compatible<NniOptimiser> for BasicParsimonyCost<A> {}

pub struct TopologyOptimiser<'a, MO, C, R>
where
    MO: MoveOptimiser,
    C: TreeSearchCost + Display + Clone + Send + Compatible<MO>,
    R: RandomSource,
{
    pub(crate) predicate: TopologyOptimiserPredicate,
    pub(crate) move_opti: MO,
    pub(crate) c: C,
    pub(crate) rng: &'a R,
}

impl<'a, MO, C, R> TopologyOptimiser<'a, MO, C, R>
where
    MO: MoveOptimiser,
    C: TreeSearchCost + Display + Clone + Send + Compatible<MO>,
    R: RandomSource,
{
    pub fn new(cost: C, move_opti: MO, rng: &'a R) -> Self {
        Self {
            predicate: TopologyOptimiserPredicate::GtEpsilon(1e-3),
            move_opti,
            c: cost,
            rng,
        }
    }

    pub fn new_with_pred(
        cost: C,
        move_opti: MO,
        rng: &'a R,
        predicate: TopologyOptimiserPredicate,
    ) -> Self {
        Self {
            c: cost,
            move_opti,
            predicate,
            rng,
        }
    }

    /// Runs the topology optimisation algorithm on the given cost function.
    /// The algorithm will iterate until the predicate is satisfied.
    /// The cost function will be updated in place.
    ///
    /// # Panics
    /// Panics if the tree has less than 4 nodes, as SPRs are not applicable to trees with less than 4 nodes.
    ///
    /// # Returns
    /// A `PhyloOptimisationResult` containing the initial cost, final cost, number of iterations, and the final cost function.
    /// The final cost function will contain the optimised tree.
    ///
    /// # Example
    /// ```rust
    /// # fn main() -> std::result::Result<(), anyhow::Error> {
    /// use phylo::likelihood::TreeSearchCost;
    /// use phylo::optimisers::{SprOptimiser, TopologyOptimiser};
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::random::DefaultGenerator;
    /// use phylo::substitution_models::{SubstModel, SubstitutionCostBuilder, K80};
    ///
    /// let info = PhyloInfoBuilder::new("./examples/data/K80.fasta").build()?;
    /// let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]);
    /// let c = SubstitutionCostBuilder::new(k80, info).build()?;
    /// let unopt_cost = c.cost();
    /// let result = TopologyOptimiser::new(c, SprOptimiser {}, &DefaultGenerator::default()).run()?;
    /// assert_eq!(unopt_cost, result.initial_cost);
    /// assert!(result.final_cost > result.initial_cost);
    /// assert!(result.iterations <= 100);
    /// assert_eq!(result.cost.tree().len(), 9); // The initial tree has 9 nodes, 5 leaves and 4 internal nodes.
    /// # Ok(()) }
    /// ```
    pub fn run(mut self) -> Result<PhyloOptimisationResult<C>> {
        info!("Optimising tree topology with SPRs");
        let init_cost = self.c.cost();
        let init_tree = self.c.tree();

        info!("Initial cost: {init_cost}");
        debug!("Initial tree: \n{init_tree}");
        let mut curr_cost = init_cost;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        let possible_move_locs: Vec<_> = self.move_opti.move_locations(&self.c).copied().collect();
        let mut current_move_locs: Vec<_> = possible_move_locs.iter().collect();

        let move_opti = self.move_opti.clone();
        // The best move on this iteration might still be worse than the current tree, in which case
        // the search stops.
        // This means that curr_cost is always higher than or equal to prev_cost.
        while self.predicate.test(iterations, curr_cost - prev_cost) {
            iterations += 1;
            info!("Iteration: {iterations}, current cost: {curr_cost}");
            prev_cost = curr_cost;

            self.rng.shuffle(&mut current_move_locs);

            curr_cost =
                Self::fold_improving_moves(&mut self.c, &move_opti, curr_cost, &current_move_locs)?;

            // Optimise branch lengths on current tree to match PhyML
            if self.c.blen_optimisation() {
                let o = BranchOptimiser::new(self.c.clone()).run()?;
                if o.final_cost > curr_cost {
                    curr_cost = o.final_cost;
                    let mut tree = o.cost.tree().clone();
                    tree.dirty();
                    self.c.update_tree(tree);
                }
            }
            debug!("Tree after iteration {}: \n{}", iterations, self.c.tree());
        }

        debug_assert_eq!(curr_cost, self.c.cost());
        info!("Done optimising tree topology");
        info!("Final cost: {curr_cost}, achieved in {iterations} iteration(s)");
        Ok(PhyloOptimisationResult {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            cost: self.c,
        })
    }

    /// Iterates over `move_locations` in order and applies the best (improving)
    /// tree move for each move location in place
    /// # Returns:
    /// - the new cost (or `base_cost` if no improvement was found)
    pub fn fold_improving_moves(
        cost_fn: &mut C,
        move_opti: &MO,
        base_cost: f64,
        move_locations: &[&NodeIdx],
    ) -> Result<f64> {
        debug_assert!(
            {
                let correct_move_locations = move_opti.move_locations(cost_fn).collect_vec();
                move_locations
                    .iter()
                    .all(|move_location| correct_move_locations.contains(move_location))
            },
            "all move locations must be contained in the tree and valid"
        );

        move_locations.iter().copied().try_fold(
            base_cost,
            |base_cost, move_location| -> Result<_> {
                let move_cost_info =
                    move_opti.best_move_at_location(base_cost, cost_fn, move_location)?;
                let MoveCostInfo {
                    cost: best_cost,
                    tree: best_tree,
                } = move_cost_info;

                if best_cost > base_cost {
                    cost_fn.update_tree(best_tree);
                    info!("    {move_opti} move applied, new cost {best_cost}");
                    Ok(best_cost)
                } else {
                    info!("    No improvement, best cost {best_cost}");
                    Ok(base_cost)
                }
            },
        )
    }
}
