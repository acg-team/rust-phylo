use std::fmt::Display;
use std::hint::black_box;
use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};

use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{spr, BranchOptimiser, ModelOptimiser};
use phylo::pip_model::{PIPCost, PIPCostBuilder, PIPModel};
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};
mod helpers;
use helpers::{
    black_box_deterministic_phylo_info, SequencePaths, AA_EASY_12X73, AA_EASY_6X97,
    DNA_EASY_5X1000, DNA_EASY_8X1252,
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

/// copied from [`TopologyOptimiser::run`]
fn fixed_iter_simulated_topo_optimiser<C: TreeSearchCost + Clone + Display>(
    mut cost_fn: C,
) -> anyhow::Result<f64> {
    let init_tree = cost_fn.tree();

    let possible_prunes: Vec<_> = init_tree.find_possible_prune_locations().copied().collect();
    let current_prunes: Vec<_> = possible_prunes.iter().collect();
    let mut curr_cost = cost_fn.cost();

    for _i in 0..3 {
        curr_cost = spr::fold_improving_moves(&mut cost_fn, f64::MIN, &current_prunes)?;

        // Optimise branch lengths on current tree to match PhyML
        let o = BranchOptimiser::new(cost_fn.clone()).run()?;
        if o.final_cost > curr_cost {
            curr_cost = o.final_cost;
            cost_fn.update_tree(o.cost.tree().clone(), &[]);
        }
    }

    Ok(curr_cost)
}

fn run_simulated_topo_for_sizes<Q: QMatrix + QMatrixMaker>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group =
        criterion.benchmark_group(format!("SIMULATED-TOPO-OPTIMISER {group_name}"));
    let mut bench = |id: &str, data: PIPCost<Q>| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                fixed_iter_simulated_topo_optimiser,
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let data = black_box_setup::<Q>(path, FrequencyOptimisation::Empirical);
        bench(key, data);
    }
    bench_group.finish();
}

fn topo_dna(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("5X1000", DNA_EASY_5X1000), ("8X1252", DNA_EASY_8X1252)]);
    run_simulated_topo_for_sizes::<JC69>(&paths, "topology optimiser DNA", criterion);
}

fn topo_aa(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("6X97", AA_EASY_6X97), ("12X73", AA_EASY_12X73)]);
    run_simulated_topo_for_sizes::<WAG>(&paths, "topology optimiser AA", criterion);
}

criterion_group! {
name = dna;
config = helpers::setup_suite().sample_size(15);
targets = topo_dna
}
criterion_group! {
name = aa;
config = helpers::setup_suite().sample_size(15);
targets = topo_aa,
}
criterion_main!(aa, dna);
