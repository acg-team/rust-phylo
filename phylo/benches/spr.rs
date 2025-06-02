use std::fmt::Display;
use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};

use itertools::Itertools;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{
    spr, RegraftOptimiser, RegraftOptimiserCacheStorageView, TopologyOptimiserStorage,
};
use phylo::pip_model::PIPCost;
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};
use phylo::tree::NodeIdx;
mod helpers;
use helpers::{
    black_box_pip_cost, SequencePaths, AA_EASY_12X73, AA_EASY_27X632, AA_EASY_6X97,
    DNA_EASY_17X2292, DNA_EASY_5X1000, DNA_EASY_8X1252,
};

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

fn run_single_spr_cycle_for_sizes<Q: QMatrix + QMatrixMaker + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(format!("SINGLE-SPR-CYCLE {group_name}"));
    let mut bench = |id: &str,
                     (mut storage, prune_locations): (
        TopologyOptimiserStorage<PIPCost<Q>>,
        &[&NodeIdx],
    )| {
        bench_group.bench_function(id, |bench| {
            bench.iter_custom(|iters| {
                let base_cost_fn = storage.base_cost_fn().clone();
                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let _ = black_box(single_spr_cycle(&mut storage, prune_locations));
                    elapsed += start.elapsed();
                    storage.set_base_cost_fn_to(&base_cost_fn);
                }
                elapsed
            });
        });
    };
    for (key, path) in paths {
        let cost_fn = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        let prune_locations = cost_fn
            .tree()
            .find_possible_prune_locations()
            .copied()
            .collect_vec();
        let prune_locations_ref = prune_locations.iter().collect_vec();
        let storage = TopologyOptimiserStorage::new_inplace(&cost_fn);
        bench(key, (storage, &prune_locations_ref));
    }
    bench_group.finish();
}

fn run_find_best_regraft_for_single_spr_move<Q: QMatrix + QMatrixMaker + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group =
        criterion.benchmark_group(format!("SINGLE-SPR-MOVE-FIND-BEST-REGRAFT {group_name}"));
    let mut bench = |id: &str, (cost_fn, prune_location): (PIPCost<Q>, _)| {
        bench_group.bench_function(id, |bench| {
            bench.iter_custom(|iters| {
                let base_cost_fn = cost_fn.clone();

                let mut elapsed = Duration::ZERO;
                let mut topo_storage = TopologyOptimiserStorage::new_inplace(&cost_fn);
                for _ in 0..iters {
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
                elapsed
            });
        });
    };
    for (key, path) in paths {
        let cost_fn = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
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

        bench(key, (cost_fn, prune_location));
    }
    bench_group.finish();
}

fn spr_dna(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("5X1000", DNA_EASY_5X1000), ("8X1252", DNA_EASY_8X1252)]);
    let long_running_paths = SequencePaths::from([("17X2292", DNA_EASY_17X2292)]);
    run_single_spr_cycle_for_sizes::<JC69>(&paths, "spr DNA", criterion);
    run_find_best_regraft_for_single_spr_move::<JC69>(&paths, "spr DNA", criterion);
    run_find_best_regraft_for_single_spr_move::<JC69>(&long_running_paths, "spr AA", criterion);
}

fn spr_aa(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("6X97", AA_EASY_6X97), ("12X73", AA_EASY_12X73)]);
    let long_running_paths = SequencePaths::from([("27X632", AA_EASY_27X632)]);
    run_single_spr_cycle_for_sizes::<WAG>(&paths, "spr AA", criterion);
    run_find_best_regraft_for_single_spr_move::<WAG>(&paths, "spr AA", criterion);
    run_find_best_regraft_for_single_spr_move::<WAG>(&long_running_paths, "spr AA", criterion);
}

criterion_group! {
name = dna;
config = helpers::setup_suite();
targets = spr_dna
}
criterion_group! {
name = aa;
config = helpers::setup_suite();
targets = spr_aa,
}
criterion_main!(aa, dna);
