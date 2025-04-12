use std::fmt::Display;
use std::hint::black_box;
use std::path::Path;
use std::result::Result::Ok;

use anyhow::Result;

use criterion::{criterion_group, criterion_main, Criterion};
use log::info;

use phylo::bench_helpers::{
    black_box_deterministic_phylo_info, SequencePaths, AA_EASY_12X73, AA_EASY_6X97,
    DNA_EASY_5X1000, DNA_EASY_8X1252,
};
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::{ModelSearchCost, TreeSearchCost};
use phylo::optimisers::{ModelOptimiser, TopologyOptimiser};
use phylo::pip_model::{PIPCost, PIPCostBuilder, PIPModel};
use phylo::substitution_models::{QMatrix, QMatrixMaker, JC69, WAG};
use phylo::tree::Tree;

#[derive(Clone)]
struct PIPConfig {
    freqs: Vec<f64>,
    params: Vec<f64>,
    freq_opt: FrequencyOptimisation,
    max_iters: usize,
    epsilon: f64,
}

fn black_box_setup<Model: QMatrix + QMatrixMaker>(
    seq_path: impl AsRef<Path>,
) -> (PIPConfig, PIPCost<Model>) {
    let info = black_box_deterministic_phylo_info(seq_path);

    let cfg = black_box(PIPConfig {
        params: vec![],
        freqs: vec![],
        freq_opt: FrequencyOptimisation::Empirical,
        epsilon: 1e-2,
        max_iters: 5,
    });

    let pip_cost = black_box(PIPCostBuilder::new(
        black_box(black_box(PIPModel::<Model>::new(&cfg.freqs, &cfg.params))),
        info,
    ))
    .build()
    .expect("failed to build pip cost optimiser");

    (cfg, pip_cost)
}

fn run_optimisation(
    cost: impl TreeSearchCost + ModelSearchCost + Display + Clone,
    freq_opt: FrequencyOptimisation,
    max_iterations: usize,
    epsilon: f64,
) -> Result<(f64, Tree)> {
    let mut cost = cost;
    let mut prev_cost = f64::NEG_INFINITY;
    let mut final_cost = TreeSearchCost::cost(&cost);

    let mut iterations = 0;
    while final_cost - prev_cost > epsilon && iterations < max_iterations {
        iterations += 1;
        info!("Iteration: {}", iterations);

        prev_cost = final_cost;
        let model_optimiser = ModelOptimiser::new(cost, freq_opt);
        let o = TopologyOptimiser::new(model_optimiser.run()?.cost)
            .run()
            .unwrap();
        final_cost = o.final_cost;
        cost = o.cost;
    }
    Ok((final_cost, cost.tree().clone()))
}

fn run_for_sizes<Q: QMatrix + QMatrixMaker>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(group_name);
    let mut bench = |id: &str, data: (PIPConfig, PIPCost<Q>)| {
        bench_group.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability in PIPCost
                || data.clone(),
                |data| run_optimisation(data.1, data.0.freq_opt, data.0.max_iters, data.0.epsilon),
                criterion::BatchSize::SmallInput,
            );
        });
    };
    for (key, path) in paths {
        let data = black_box_setup::<Q>(path);
        bench(key, data);
    }
    bench_group.finish();
}

fn pip_inferrence_dna(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("5X1000", DNA_EASY_5X1000), ("8X1252", DNA_EASY_8X1252)]);
    run_for_sizes::<JC69>(&paths, "Tree-from-MSA DNA", criterion);
}

fn pip_inferrence_aa(criterion: &mut Criterion) {
    let paths = SequencePaths::from([("6X97", AA_EASY_6X97), ("12X73", AA_EASY_12X73)]);
    run_for_sizes::<WAG>(&paths, "Tree-from-MSA AA", criterion);
}

criterion_group! {
name = dna;
config = Criterion::default().sample_size(10);
targets = pip_inferrence_dna
}
criterion_group! {
name = aa;
config = Criterion::default().sample_size(10);
targets = pip_inferrence_aa,
}
criterion_main!(aa, dna);
