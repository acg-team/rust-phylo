use std::fmt::Display;
use std::hint::black_box;
use std::path::PathBuf;
use std::result::Result::Ok;
use std::time::Duration;

use anyhow::Result;

use criterion::{criterion_group, criterion_main, Criterion};
use log::info;

use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::{ModelSearchCost, TreeSearchCost};
use phylo::optimisers::{ModelOptimiser, TopologyOptimiser};
use phylo::phylo_info::PhyloInfoBuilder;
use phylo::pip_model::{PIPCost, PIPCostBuilder, PIPModel};
use phylo::substitution_models::{QMatrix, QMatrixMaker, GTR, WAG};
use phylo::tree::Tree;

struct PIPConfig {
    freqs: Vec<f64>,
    params: Vec<f64>,
    freq_opt: FrequencyOptimisation,
    max_iters: usize,
    epsilon: f64,
}

fn black_box_setup<Model: QMatrix + QMatrixMaker>(
    path: impl Into<PathBuf>,
) -> (PIPConfig, PIPCost<Model>) {
    let seq_file = black_box(path.into());

    let info = black_box(
        PhyloInfoBuilder::new(seq_file)
            .tree_file(None)
            .build()
            .expect("failed to build phylo info"),
    );

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

fn bench_pip_protein_small(criterion: &mut criterion::Criterion) {
    let (cfg, pip_cost) = black_box_setup::<WAG>("data/benches/single-gene/NagyA1/Cluster9992.aln");
    criterion.bench_function("PIP Protein(WAG) small", |bench| {
        bench.iter(|| {
            run_optimisation(pip_cost.clone(), cfg.freq_opt, cfg.max_iters, cfg.epsilon)
                .expect("failed to run pip optimisation")
        });
    });
}
fn bench_pip_dna_tiny(criterion: &mut criterion::Criterion) {
    let (cfg, pip_cost) = black_box_setup::<GTR>("data/sim/GTR/gtr.fasta");
    criterion.bench_function("PIP DNA(GTR) tiny but long", |bench| {
        bench.iter(|| {
            run_optimisation(pip_cost.clone(), cfg.freq_opt, cfg.max_iters, cfg.epsilon)
                .expect("failed to run pip optimisation")
        });
    });
}

fn run_optimisation(
    cost: impl TreeSearchCost + ModelSearchCost + Display + Clone + Send,
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

criterion_group! {
name = pip_inferrence_tiny;
config = Criterion::default().measurement_time(Duration::from_secs(30)).sample_size(10);
targets = bench_pip_dna_tiny
}
criterion_group! {
name = pip_inferrence_small;
config = Criterion::default().measurement_time(Duration::from_secs(20)).sample_size(10);
targets = bench_pip_protein_small
}
criterion_main!(pip_inferrence_tiny, pip_inferrence_small);
