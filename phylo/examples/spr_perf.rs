use std::fmt::Display;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use itertools::Itertools;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{
    spr, RegraftOptimiser, RegraftOptimiserCacheStorageView, TopologyOptimiserStorage,
};
use phylo::pip_model::PIPCost;
use phylo::substitution_models::{QMatrix, QMatrixMaker, GTR};
use phylo::tree::NodeIdx;

use phylo::{
    optimisers::ModelOptimiser,
    phylo_info::{PhyloInfo, PhyloInfoBuilder},
    pip_model::{PIPCostBuilder, PIPModel},
};

const ITERS: usize = 1000;
const DNA_EASY_17X2292: &str = "data/benchmark-datasets/dna/easy/wickd3a_7771.processed.fasta";

fn black_box_deterministic_phylo_info(seq_file: impl Into<PathBuf>) -> PhyloInfo {
    assert!(
        cfg!(feature = "deterministic"),
        "only run benches with '-F deterministic'"
    );
    black_box(
        PhyloInfoBuilder::new(seq_file.into())
            .build()
            .expect("sequence file should be able to build phylo info"),
    )
}
fn black_box_pip_cost<Model: QMatrix + QMatrixMaker>(
    path: impl Into<PathBuf>,
    freq_opt: FrequencyOptimisation,
) -> PIPCost<Model> {
    let info = black_box_deterministic_phylo_info(path);
    let pip_cost = PIPCostBuilder::new(PIPModel::<Model>::new(&[], &[]), info)
        .build()
        .expect("failed to build pip cost optimiser");

    // done for a more 'realistic' setup
    let model_optimiser = ModelOptimiser::new(pip_cost, freq_opt);
    black_box(
        model_optimiser
            .run()
            .expect("model optimiser should pass")
            .cost,
    )
}

fn single_spr_cycle<C: TreeSearchCost + Clone + Display + Send>(
    storage: &mut TopologyOptimiserStorage<C>,
    prune_locations: &[&NodeIdx],
) -> anyhow::Result<f64> {
    spr::fold_improving_moves(storage, f64::MIN, prune_locations)
}

fn find_best_regraft_for_single_spr_move<'a, 'b: 'a, C: TreeSearchCost + Clone + Display + Send>(
    regraft_optimiser: &'a mut RegraftOptimiser<'b, C, RegraftOptimiserCacheStorageView<'b, C>>,
) -> anyhow::Result<f64> {
    let best_regraft = regraft_optimiser
        .find_max_cost_regraft_for_prune(f64::MIN)?
        .expect("invalid prune location for benchmarking");
    Ok(best_regraft.cost())
}

fn run_single_spr_cycle_for_sizes() {
    let cost_fn = black_box_pip_cost::<GTR>(DNA_EASY_17X2292, FrequencyOptimisation::Empirical);
    let prune_locations = cost_fn
        .tree()
        .find_possible_prune_locations()
        .copied()
        .collect_vec();
    let prune_locations = prune_locations.iter().collect_vec();

    let mut storage = TopologyOptimiserStorage::new_inplace(&cost_fn);
    let base_cost_fn = storage.base_cost_fn().clone();
    let mut elapsed = Duration::ZERO;
    for _ in 0..ITERS {
        let start = Instant::now();
        let _ = black_box(single_spr_cycle(&mut storage, &prune_locations));
        elapsed += start.elapsed();
        storage.set_cost_fns_to(&base_cost_fn);
    }
}

fn run_find_best_regraft_for_single_spr_move() {
    let cost_fn = black_box_pip_cost::<GTR>(DNA_EASY_17X2292, FrequencyOptimisation::Empirical);
    let tree = cost_fn.tree();
    // NOTE: regrafting an early preorder node would mean that a long path along the tree stays in tact
    // and less has to be re-calculated overall. We try to benchmark a likely worst case
    // since all parents have to be re-calculated
    let prune_location = *tree
        .postorder()
        .iter()
        .filter(|&n| n != &tree.root)
        .find(|prune| !cost_fn.tree().node(&tree.root).children.contains(prune))
        .expect("tree should have at least one node not a direct child of root");
    let base_cost_fn = cost_fn.clone();

    let mut elapsed = Duration::ZERO;
    let mut topo_storage = TopologyOptimiserStorage::new_inplace(&cost_fn);
    for _ in 0..ITERS {
        let mut regraft_optimiser = RegraftOptimiser::new_with_storage(
            &prune_location,
            RegraftOptimiserCacheStorageView::new(&mut topo_storage),
        );
        let start = Instant::now();
        let _ = black_box(find_best_regraft_for_single_spr_move(
            &mut regraft_optimiser,
        ));
        elapsed += start.elapsed();
        topo_storage.set_base_cost_fn_to(&base_cost_fn);
    }
}

fn main() {
    run_find_best_regraft_for_single_spr_move();
}
