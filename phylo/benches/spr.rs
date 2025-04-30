use std::fmt::Display;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};

use itertools::Itertools;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{ModelOptimiser, TopologyOptimiser};
use phylo::pip_model::{PIPCost, PIPCostBuilder, PIPModel};
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};
use phylo::tree::NodeIdx;
mod helpers;
use helpers::{
    black_box_deterministic_phylo_info, SequencePaths, AA_EASY_12X73, AA_EASY_27X632, AA_EASY_6X97,
    DNA_EASY_17X2292, DNA_EASY_5X1000, DNA_EASY_8X1252,
};

fn black_box_setup<Model: QMatrix + QMatrixMaker>(
    path: impl Into<PathBuf>,
    freq_opt: FrequencyOptimisation,
) -> PIPCost<Model> {
    let info = black_box_deterministic_phylo_info(path);
    let pip_cost = PIPCostBuilder::new(PIPModel::<Model>::new(&[], &[]), info)
        .build()
        .expect("failed to build pip cost optimiser");

    // TODO: don't know if this is necessary but since the JATI repo calls this before running the
    // TopoOptimiser I think its more accurate to also do it here
    let model_optimiser = ModelOptimiser::new(pip_cost, freq_opt);
    black_box(
        model_optimiser
            .run()
            .expect("model optimiser should pass")
            .cost,
    )
}

fn single_spr_cycle<C: TreeSearchCost + Clone + Display>(
    mut cost_fn: C,
    prune_locations: &[&NodeIdx],
) -> anyhow::Result<f64> {
    TopologyOptimiser::fold_improving_spr_moves(&mut cost_fn, f64::MIN, prune_locations)
}

fn find_best_regraft_for_single_spr_move<C: TreeSearchCost + Clone + Display>(
    cost_fn: C,
    prune_location: &NodeIdx,
) -> anyhow::Result<f64> {
    let best_regraft =
        TopologyOptimiser::find_max_cost_regraft_for_prune(prune_location, f64::MIN, &cost_fn)?
            .expect("invalid prune location for benchmarking");
    Ok(best_regraft.cost)
}

fn run_single_spr_cycle_for_sizes<Q: QMatrix + QMatrixMaker>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(format!("SINGLE-SPR-CYCLE {group_name}"));
    let mut bench = |id: &str, data: (PIPCost<Q>, &[&NodeIdx])| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                |(cost_fn, prune_locations)| single_spr_cycle(cost_fn, prune_locations),
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let cost_fn = black_box_setup::<Q>(path, FrequencyOptimisation::Empirical);
        let prune_locations = cost_fn
            .tree()
            .find_possible_prune_locations()
            .copied()
            .collect_vec();
        let prune_locations_ref = prune_locations.iter().collect_vec();
        bench(key, (cost_fn, &prune_locations_ref));
    }
    bench_group.finish();
}

fn run_find_best_regraft_for_single_spr_move<Q: QMatrix + QMatrixMaker>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group =
        criterion.benchmark_group(format!("SINGLE-SPR-MOVE-FIND-BEST-REGRAFT {group_name}"));
    let mut bench = |id: &str, data: (PIPCost<Q>, &NodeIdx)| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                |(cost_fn, prune_locations)| {
                    find_best_regraft_for_single_spr_move(cost_fn, prune_locations)
                },
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let cost_fn = black_box_setup::<Q>(path, FrequencyOptimisation::Empirical);
        let tree = cost_fn.tree();
        // NOTE: regrafting an early preorder node would mean that a long branch stays in tact
        // and less has to be re-calculated overall. We try to benchmark a likely worst case
        // since all parents have to be re-calculated
        let prune_location = *tree
            .postorder()
            .iter()
            .filter(|&n| n != &tree.root)
            .find(|prune| !cost_fn.tree().node(&tree.root).children.contains(prune))
            .expect("tree should have at least one node not a direct child of root");
        bench(key, (cost_fn, &prune_location));
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
