use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::DerefMut;

use itertools::Itertools;
use log::{debug, info};

use crate::likelihood::TreeSearchCost;
use crate::optimisers::BranchOptimiser;
use crate::tree::{NodeIdx, Tree};
use crate::Result;

use super::max_regrafts_for_tree;

pub struct RegraftCostInfo {
    regraft: NodeIdx,
    cost: f64,
    tree: Tree,
}

impl RegraftCostInfo {
    pub fn regraft(&self) -> NodeIdx {
        self.regraft
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

pub trait RegraftOptimiserStorage<'a, C: TreeSearchCost + Clone + Display + Send> {
    fn base(&self) -> &C;
    fn cost_fns_mut(&mut self) -> &mut [C];
}

pub struct RegraftOptimiser<
    'a,
    C: TreeSearchCost + Clone + Display + Send,
    S: RegraftOptimiserStorage<'a, C>,
> {
    phantom: PhantomData<C>,
    prune_location: &'a NodeIdx,
    storage: S,
}
pub struct RegraftOptimiserSimpleStorage<'a, C: TreeSearchCost + Clone + Display + Send> {
    cost_fns: Box<[C]>,
    // TODO: replace with first
    cost_fn: &'a C,
}

impl<'a, C: TreeSearchCost + Clone + Display + Send> RegraftOptimiserSimpleStorage<'a, C> {
    pub fn new(cost_fn: &'a C) -> RegraftOptimiserSimpleStorage<'a, C> {
        let max_regrafts = max_regrafts_for_tree(cost_fn.tree());
        Self {
            cost_fns: vec![cost_fn.clone(); max_regrafts].into_boxed_slice(),
            cost_fn,
        }
    }
}
impl<'a, C: TreeSearchCost + Clone + Display + Send> RegraftOptimiserStorage<'a, C>
    for RegraftOptimiserSimpleStorage<'a, C>
{
    fn base(&self) -> &C {
        self.cost_fn
    }

    fn cost_fns_mut(&mut self) -> &mut [C] {
        &mut self.cost_fns
    }
}

pub struct RegraftOptimiserCacheStorageView<'a, C: TreeSearchCost + Clone + Display + Send> {
    cost_fns: &'a mut [C],
}

impl<'a, C: TreeSearchCost + Clone + Display + Send> RegraftOptimiserStorage<'a, C>
    for RegraftOptimiserCacheStorageView<'a, C>
{
    fn base(&self) -> &C {
        &self.cost_fns[0]
    }
    fn cost_fns_mut(&mut self) -> &mut [C] {
        self.cost_fns.deref_mut()
    }
}

impl<'a, C: TreeSearchCost + Clone + Display + Send> RegraftOptimiserCacheStorageView<'a, C> {
    pub fn new(cost_fns: &'a mut [C]) -> RegraftOptimiserCacheStorageView<'a, C> {
        Self { cost_fns }
    }
}
impl<'a, C: TreeSearchCost + Clone + Display + Send + 'a, S: RegraftOptimiserStorage<'a, C>>
    RegraftOptimiser<'a, C, S>
{
    pub fn new(
        cost_fn: &'a C,
        prune_location: &'a NodeIdx,
    ) -> RegraftOptimiser<'a, C, RegraftOptimiserSimpleStorage<'a, C>> {
        RegraftOptimiser {
            phantom: PhantomData,
            prune_location,
            storage: RegraftOptimiserSimpleStorage::new(cost_fn),
        }
    }
    pub fn new_with_storage(prune_location: &'a NodeIdx, storage: S) -> RegraftOptimiser<'a, C, S> {
        Self {
            phantom: PhantomData,
            prune_location,
            storage,
        }
    }

    pub fn available_regraft_locations(&self) -> impl Iterator<Item = &NodeIdx> + use<'_, C, S> {
        let tree = self.storage.base().tree();
        let all_locations = tree.preorder();
        let prune_subtrees = tree.preorder_subroot(self.prune_location);
        let sibling = tree.sibling(self.prune_location).unwrap();
        let parent = tree.node(self.prune_location).parent.unwrap();
        all_locations.iter().filter(move |&node| {
            *node != sibling
                && *node != parent
                && *node != tree.root
                && !prune_subtrees.contains(node)
        })
    }

    // TODO/MERBUG this should probably only be called once or the cache needs cleaning
    pub fn find_max_cost_regraft_for_prune(
        &mut self,
        base_cost: f64,
    ) -> Result<Option<RegraftCostInfo>> {
        let tree = self.storage.base().tree();
        if tree.children(&tree.root).contains(self.prune_location) {
            // due to topology change the current node may have become the direct child of root
            return Ok(None);
        }

        let regraft_locations = self.available_regraft_locations().copied().collect_vec();

        let cost_fns = &mut self.storage.cost_fns_mut()[..regraft_locations.len()];

        info!("Node {:?}: trying to regraft", self.prune_location);
        let best_regraft =
            calc_best_regraft_cost(base_cost, *self.prune_location, regraft_locations, cost_fns)?;
        Ok(Some(best_regraft))
    }
}

cfg_if::cfg_if! {
if #[cfg(feature="par-regraft")] {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost_fns: &mut [C],
) -> Result<RegraftCostInfo> {
    use rayon::prelude::*;
    regraft_locations
        .into_par_iter()
        .zip_eq(cost_fns)
        .map(move |(regraft, cost_fn)| {
            calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost_fn)
        })
        .try_reduce_with(|left, right| Ok(if left.cost() > right.cost() {left} else {right})).expect("at least one regraft location")
}
} else if #[cfg(feature="par-regraft-chunk")] {
/// NOTE: seems to be faster than full on parallel for few taxa
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost_fns: &mut [C],
) -> Result<RegraftCostInfo> {
    use rayon::prelude::*;
    // TODO: determine better factor (maybe dynamically)
    const CHUNK_SIZE: usize = 2;
    let n_regrafts = regraft_locations.len();
    let n_chunks = n_regrafts.div_ceil(CHUNK_SIZE);

    let mut cost_fn_chunks = Vec::with_capacity(n_chunks);

    let mut cost_fns = cost_fns;
    for i in (0..regraft_locations.len()).step_by(CHUNK_SIZE) {
        // not divisible by chunk size
        if n_regrafts - i < CHUNK_SIZE {
            cost_fn_chunks.push(cost_fns);
            break;
        } else {
            let (chunk, rest) = cost_fns.split_at_mut(CHUNK_SIZE);
            cost_fn_chunks.push(chunk);
            cost_fns = rest;
        }
    }


    regraft_locations
        .par_chunks(CHUNK_SIZE)
        .zip_eq(cost_fn_chunks)
        .map(move |(regrafts, cost_fn_chunk)| -> Result<_> {
            let mut max: Option<RegraftCostInfo> = None;
            let mut max_cost = f64::MIN;
            for regraft_result in regrafts.iter().zip_eq(cost_fn_chunk).map(move |(regraft, cost_fn)| {
                calc_spr_cost_with_blen_opt(prune_location, *regraft, base_cost, cost_fn)
            }) {
                match regraft_result {
                    Ok(regraft_info) if regraft_info.cost() > max_cost => {
                        max_cost = regraft_info.cost();
                        max = Some(regraft_info);
                    },
                    Ok(_) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(max.expect("at least one regraft location"))
        })
        .try_reduce_with(|left, right| Ok(if left.cost() > right.cost() {left} else {right})).expect("at least one regraft location")
}
} else if #[cfg(feature="par-regraft-manual")] {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost_fns: &mut [C],
) -> Result<RegraftCostInfo> {
    #[derive(Clone, Copy)]
    struct RecursiveForkJoinRegrafter {
        prune_location: NodeIdx,
        base_cost: f64,
    }
    /// NOTE: by being recursive these tasks can be stored solely on the stack
    /// using rayon::scope might look simpler but incurrs overhead by having to manage
    /// tasks on the heap
    fn regraft_recursive<C: TreeSearchCost + Clone + Display + Send>(state: RecursiveForkJoinRegrafter, regraft_locations: &[NodeIdx], cost_fns: &mut [C]) -> Result<RegraftCostInfo> {
        debug_assert_eq!(regraft_locations.len(), cost_fns.len());
        if regraft_locations.len() == 1 {
            return calc_spr_cost_with_blen_opt(state.prune_location, regraft_locations[0], state.base_cost, &mut cost_fns[0]);
        }
        let mid =regraft_locations.len() / 2;
        let (left_cost_fns, right_cost_fns) = cost_fns.split_at_mut(mid);
        let (left_locations, right_locations) = regraft_locations.split_at(mid);
        match rayon::join(move || regraft_recursive(state, left_locations, left_cost_fns), move ||regraft_recursive(state, right_locations, right_cost_fns)) {
            (Ok(left), Ok(right)) => Ok(if left.cost() > right.cost() {left} else {right}) ,
            (Err(error), _) | (_, Err(error))   => Err(error),
        }
    }
    regraft_recursive(RecursiveForkJoinRegrafter { prune_location, base_cost }, &regraft_locations, cost_fns)
}
} else {
fn calc_best_regraft_cost<C: TreeSearchCost + Clone + Display + Send>(
    base_cost: f64,
    prune_location: NodeIdx,
    regraft_locations: Vec<NodeIdx>,
    cost_fns: &mut [C],
) -> Result<RegraftCostInfo> {
    let mut max = None;
    let mut max_cost = f64::MIN;
    for regraft in itertools::zip_eq(regraft_locations, cost_fns).map(move |(regraft, cost_fn)| {
        calc_spr_cost_with_blen_opt(prune_location, regraft, base_cost, cost_fn)
    }) {
        match regraft {
            Ok(regraft_info) if regraft_info.cost() > max_cost => {
                max_cost = regraft_info.cost();
                max = Some(regraft_info);
            },
            Ok(_) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(max.expect("at least one regraft location"))
}
}}

/// for evo models with branch length optimisation enabled (disabled for parsimony),
/// if the move doesn't result in improvement over `base_cost`
/// the blen of the regrafted branch is optimised to check if an
/// improvement could still be reached
fn calc_spr_cost_with_blen_opt<C: TreeSearchCost + Clone + Display>(
    prune_location: NodeIdx,
    regraft: NodeIdx,
    base_cost: f64,
    cost_func: &mut C,
) -> Result<RegraftCostInfo> {
    let mut new_tree = cost_func.tree().rooted_spr(&prune_location, &regraft)?;

    cost_func.update_tree(new_tree.clone(), &[prune_location, regraft]);

    let mut move_cost = cost_func.cost();
    if cost_func.blen_optimisation() && move_cost <= base_cost {
        // reoptimise branch length at the regraft location
        let mut o = BranchOptimiser::new(cost_func.clone());
        let blen_opt = o.optimise_branch(&regraft)?;
        if blen_opt.final_cost > move_cost {
            move_cost = blen_opt.final_cost;
            new_tree.set_blen(&regraft, blen_opt.value);
        }
    }
    debug!("    Regraft to {:?} w best cost {}.", regraft, move_cost);
    Ok(RegraftCostInfo {
        cost: move_cost,
        regraft,
        tree: new_tree,
    })
}
