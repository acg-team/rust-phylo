use std::hint::black_box;
use std::num::NonZero;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, Criterion};

use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::optimisers::{TopologyOptimiser, TopologyOptimiserPredicate};
use phylo::pip_model::PIPCost;
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};
mod helpers;
use helpers::{
    black_box_pip_cost, SequencePaths, AA_EASY_12X73, AA_EASY_6X97, DNA_EASY_5X1000,
    DNA_EASY_8X1252,
};

fn run_fixed_iter_topo<Q: QMatrix>(
    topo_opt: &mut TopologyOptimiser<PIPCost<Q>>,
) -> anyhow::Result<f64> {
    Ok(topo_opt.run_mut()?.final_cost)
}

fn run_simulated_topo_for_sizes<Q: QMatrix + QMatrixMaker + Send>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group =
        criterion.benchmark_group(format!("SIMULATED-TOPO-OPTIMISER {group_name}"));
    let mut bench = |id: &str, mut topo_opt: TopologyOptimiser<PIPCost<Q>>| {
        bench_group.bench_function(id, |bench| {
            bench.iter_custom(|iters| {
                let base_clone = topo_opt.base_cost_fn().clone();
                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let _ = black_box(run_fixed_iter_topo(&mut topo_opt));
                    elapsed += start.elapsed();
                    topo_opt.set_base_cost_fn_to(&base_clone);
                }
                elapsed
            });
        });
    };
    for (key, path) in paths {
        let cost = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);

        let topo_opt = TopologyOptimiser::new_with_pred(
            cost,
            TopologyOptimiserPredicate::fixed_iter(NonZero::new(3).unwrap()),
        );
        bench(key, topo_opt);
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
