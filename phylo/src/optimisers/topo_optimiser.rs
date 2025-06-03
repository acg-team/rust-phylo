use std::fmt::Display;
use std::num::NonZeroUsize;

use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, PhyloOptimisationResult};
use crate::pip_model::PIPCost;
use crate::substitution_models::QMatrix;
use crate::tree::Tree;
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

// TODO: MERBUG fix this in thesis
pub(crate) fn max_regrafts_for_tree(tree: &Tree) -> usize {
    tree.nodes.len() - 4
}

mod storage {
    use rayon::iter::ParallelIterator;
    use rayon::slice::ParallelSliceMut;

    use crate::likelihood::TreeSearchCost;
    use crate::optimisers::max_regrafts_for_tree;
    use crate::pip_model::PIPCost;
    use crate::substitution_models::QMatrix;
    use crate::util::mem::boxed::BoxSlice;
    use std::fmt::Display;
    use std::mem::MaybeUninit;
    use std::num::NonZero;
    use std::ops::DerefMut;
    use std::panic;
    use std::ptr::NonNull;
    pub struct TopologyOptimiserStorage<C: TreeSearchCost + Display + Clone + Send> {
        cost_fns: Box<[C]>,
        // safety: don't read/write the value, possible race condition
        buf_base: Option<NonNull<[f64]>>,
        valid_mut_cost_fns: usize,
    }

    impl<C: TreeSearchCost + Display + Clone + Send> TopologyOptimiserStorage<C> {
        pub fn new_basic(base_cost_fn: C) -> Self {
            let total_cost_fns = super::max_regrafts_for_tree(base_cost_fn.tree()) + 1;
            Self {
                buf_base: None,
                cost_fns: vec![base_cost_fn; total_cost_fns].into_boxed_slice(),
                valid_mut_cost_fns: total_cost_fns - 1,
            }
        }
        pub fn base_cost_fn(&self) -> &C {
            self.cost_fns.first().expect("always at least one cost fn")
        }
        pub fn base_cost_fn_mut(&mut self) -> &mut C {
            self.valid_mut_cost_fns = 0;
            self.cost_fns
                .first_mut()
                .expect("always at least one cost fn")
        }
        pub fn cost_fns_mut(&mut self) -> &mut [C] {
            assert!(self.valid_mut_cost_fns == self.cost_fns.len());
            self.valid_mut_cost_fns = 0;
            &mut self.cost_fns[1..]
        }
        pub fn cost_fns_mut_up_to_excluding(&mut self, to: usize) -> &mut [C] {
            assert!(self.valid_mut_cost_fns >= to);
            self.valid_mut_cost_fns = 0;
            &mut self.cost_fns[1..to + 1]
        }

        // NOTE: could be optimized by clearing buf once and then simply
        // clearing all *valid flags individually
        pub fn set_cost_fns_to_base_upto_excluding(&mut self, to: usize) {
            assert!(to < self.cost_fns.len());
            if self.valid_mut_cost_fns >= to {
                return;
            }
            self.valid_mut_cost_fns = to;
            let n_cost_fns = self.cost_fns.len();
            if let Some(mut buf_base) = self.buf_base {
                let single_len = buf_base.len() / n_cost_fns;

                let buf_slice = unsafe { buf_base.as_mut() };

                let (ref base, mut_cost_fns) = buf_slice.split_at_mut(single_len);

                let mut par_copy = mut_cost_fns[..single_len * to].par_chunks_exact_mut(single_len);
                assert_eq!(par_copy.remainder().len(), 0);
                par_copy.for_each(|cost_fn_buf| cost_fn_buf.copy_from_slice(base));
            }
            let [ref base, others @ ..] = self.cost_fns.deref_mut() else {
                panic!("always at least one cost function in storage")
            };
            others[..to]
                .iter_mut()
                .for_each(|cost_fn| cost_fn.clone_from_smart(base));
        }
        pub fn set_base_cost_fn_to(&mut self, new_base: &C) {
            self.valid_mut_cost_fns = 0;
            self.base_cost_fn_mut().clone_from(new_base);
        }
    }

    impl<C: TreeSearchCost + Display + Clone + Send> Drop for TopologyOptimiserStorage<C> {
        fn drop(&mut self) {
            if let Some(mut owned_buf_ptr) = self.buf_base {
                drop(unsafe { BoxSlice::from_raw(owned_buf_ptr.as_mut()) });
            }
        }
    }

    impl<Q: QMatrix> TopologyOptimiserStorage<PIPCost<Q>> {
        pub fn new_inplace(cost_fn: &PIPCost<Q>) -> Self {
            let single_dimensions = cost_fn.cache_dimensions();
            let total_cost_fns = max_regrafts_for_tree(cost_fn.tree()) + 1;
            let total_padded_len = total_cost_fns * single_dimensions.total_len_f64_padded();

            // safety: gets dealloc with BoxSlice in Drop
            let buf = unsafe {
                BoxSlice::alloc_slice_uninit(NonZero::try_from(total_padded_len).unwrap()).leak()
            };
            let buf_base = NonNull::new(buf).unwrap();

            let mut ranges = Vec::with_capacity(total_cost_fns);
            let mut buf_mut = buf;
            for _ in 0..total_cost_fns {
                let (sub_slice, rest) =
                    buf_mut.split_at_mut(single_dimensions.total_len_f64_padded());
                ranges.push(sub_slice);
                buf_mut = rest;
            }
            assert_eq!(buf_mut.len(), 0);

            // safety: we manually drop the full buf
            let ranges = unsafe {
                std::mem::transmute::<
                    &mut [&mut [MaybeUninit<f64>]],
                    &mut [&'static mut [MaybeUninit<f64>]],
                >(ranges.as_mut_slice())
            };

            Self {
                valid_mut_cost_fns: total_cost_fns - 1,
                cost_fns: ranges
                    .iter_mut()
                    .map(|sub_slice| cost_fn.clone_with_cache_in(sub_slice))
                    .collect(),
                // safety: after initialization of all sub_slices
                buf_base: unsafe {
                    Some(std::mem::transmute::<
                        NonNull<[MaybeUninit<f64>]>,
                        NonNull<[f64]>,
                    >(buf_base))
                },
            }
        }
    }
}
pub use storage::TopologyOptimiserStorage;

use super::PhyloOptimisationResultStats;

pub struct TopologyOptimiser<C: TreeSearchCost + Display + Clone + Send> {
    pub(crate) predicate: TopologyOptimiserPredicate,
    pub(crate) storage: TopologyOptimiserStorage<C>,
}
impl<Q: QMatrix> TopologyOptimiser<PIPCost<Q>> {
    pub fn set_base_cost_fn_to(&mut self, new_base: &PIPCost<Q>) {
        self.storage.set_base_cost_fn_to(new_base);
    }
    pub fn new_with_pred_inplace(cost: &PIPCost<Q>, predicate: TopologyOptimiserPredicate) -> Self {
        Self {
            predicate,
            storage: TopologyOptimiserStorage::new_inplace(cost),
        }
    }
}

impl<C: TreeSearchCost + Clone + Display + Send> TopologyOptimiser<C> {
    pub fn new(cost: C) -> Self {
        Self::new_with_pred(cost, TopologyOptimiserPredicate::GtEpsilon(1e-3))
    }
    pub fn new_with_pred(cost: C, predicate: TopologyOptimiserPredicate) -> Self {
        Self {
            predicate,
            storage: TopologyOptimiserStorage::new_basic(cost),
        }
    }

    pub fn base_cost_fn(&self) -> &C {
        self.storage.base_cost_fn()
    }
    pub fn base_cost_fn_mut(&mut self) -> &mut C {
        self.storage.base_cost_fn_mut()
    }

    pub fn run(mut self) -> Result<PhyloOptimisationResult<C>> {
        let PhyloOptimisationResultStats {
            initial_cost,
            final_cost,
            iterations,
        } = self.run_mut()?;

        Ok(PhyloOptimisationResult {
            initial_cost,
            final_cost,
            iterations,
            cost: self.base_cost_fn().clone(),
        })
    }
    pub fn run_mut(&mut self) -> Result<PhyloOptimisationResultStats> {
        debug_assert!(self.storage.base_cost_fn().tree().len() > 3);

        info!("Optimising tree topology with SPRs.");
        let init_cost = self.storage.base_cost_fn().cost();
        let init_tree = self.storage.base_cost_fn().tree();

        info!("Initial cost: {}.", init_cost);
        debug!("Initial tree: \n{}", init_tree);
        let mut curr_cost = init_cost;
        let mut prev_cost = f64::NEG_INFINITY;
        let mut iterations = 0;

        let possible_prunes: Vec<_> = init_tree.find_possible_prune_locations().copied().collect();
        let current_prunes: Vec<_> = possible_prunes.iter().collect();
        cfg_if::cfg_if! {
        if #[cfg(not(feature = "deterministic"))] {
            let mut current_prunes = current_prunes;
            // TODO: decide on an explicit and consistent RNG to use throughout the project
            let rng = &mut rand::thread_rng();
        }
        }

        // The best move on this iteration might still be worse than the current tree, in which case
        // the search stops.
        // This means that curr_cost is always hugher than or equel to prev_cost.
        while self.predicate.test(iterations, curr_cost - prev_cost) {
            iterations += 1;
            info!("Iteration: {}, current cost: {}.", iterations, curr_cost);
            prev_cost = curr_cost;

            #[cfg(not(feature = "deterministic"))]
            {
                use rand::seq::SliceRandom;
                current_prunes.shuffle(rng);
            }

            curr_cost = spr::fold_improving_moves(&mut self.storage, curr_cost, &current_prunes)?;

            // Optimise branch lengths on current tree to match PhyML
            if self.storage.base_cost_fn().blen_optimisation() {
                self.storage.set_cost_fns_to_base_upto_excluding(1);
                let [branch_opt_cost] = self.storage.cost_fns_mut_up_to_excluding(1) else {
                    panic!("requesting one cost_fn returns one");
                };
                let o = BranchOptimiser::new(branch_opt_cost).run()?;
                if o.final_cost > curr_cost {
                    curr_cost = o.final_cost;
                    self.storage
                        .base_cost_fn_mut()
                        .update_tree(o.cost.tree().clone(), &[]);
                }
            }
            debug!(
                "Tree after iteration {}: \n{}",
                iterations,
                self.storage.base_cost_fn().tree()
            );
        }

        debug_assert_eq!(curr_cost, self.storage.base_cost_fn().cost());
        info!("Done optimising tree topology.");
        info!(
            "Final cost: {}, achieved in {} iteration(s).",
            curr_cost, iterations
        );
        Ok(PhyloOptimisationResultStats {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
        })
    }
}

pub mod spr {
    use std::fmt::Display;

    use itertools::Itertools;
    use log::info;

    use crate::{
        likelihood::TreeSearchCost,
        optimisers::{RegraftOptimiser, RegraftOptimiserCacheStorageView},
        tree::NodeIdx,
        Result,
    };

    use super::TopologyOptimiserStorage;

    /// Iterates over `prune_locations` in order and applies the best (improving)
    /// SPR move for each pruneing location in place
    /// # Returns:
    /// - the new cost (or `base_cost` if no improvement was found)
    pub fn fold_improving_moves<C: TreeSearchCost + Display + Clone + Send>(
        storage: &mut TopologyOptimiserStorage<C>,
        base_cost: f64,
        prune_locations: &[&NodeIdx],
    ) -> Result<f64> {
        debug_assert!(
            {
                let correct_prune_locations = storage
                    .base_cost_fn()
                    .tree()
                    .find_possible_prune_locations()
                    .collect_vec();
                prune_locations
                    .iter()
                    .all(|prune_location| correct_prune_locations.contains(prune_location))
            },
            "all prune locations must be contained in the tree and valid"
        );

        prune_locations
            .iter()
            .copied()
            .try_fold(base_cost, |base_cost, prune| -> Result<_> {
                let mut regraft_optimiser = RegraftOptimiser::new_with_storage(
                    prune,
                    RegraftOptimiserCacheStorageView::new(storage),
                );

                let Some(best_regraft_info) =
                    regraft_optimiser.find_max_cost_regraft_for_prune(base_cost)?
                else {
                    return Ok(base_cost);
                };

                let (best_cost, best_regraft, best_tree) = (
                    best_regraft_info.cost(),
                    best_regraft_info.regraft(),
                    best_regraft_info.into_tree(),
                );

                let new_cost = if best_cost > base_cost {
                    storage
                        .base_cost_fn_mut()
                        .update_tree(best_tree, &[*prune, best_regraft]);
                    info!(
                        "    Regrafted to {:?}, new cost {}.",
                        best_regraft, best_cost
                    );
                    best_cost
                } else {
                    info!("    No improvement, best cost {}.", best_cost);
                    base_cost
                };

                Ok(new_cost)
            })
    }
}
