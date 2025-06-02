use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{criterion_group, criterion_main, Criterion};
use phylo::{
    evolutionary_models::FrequencyOptimisation,
    likelihood::ModelSearchCost,
    pip_model::PIPCost,
    substitution_models::{QMatrix, QMatrixMaker, JC69, WAG},
    util::mem::boxed::BoxSlice,
};
mod helpers;
use helpers::{
    SequencePaths, AA_EASY_12X73, AA_EASY_14X165, AA_EASY_27X632, AA_EASY_45X223, AA_EASY_6X97,
    AA_MEDIUM_79X106, DNA_EASY_17X2292, DNA_EASY_33X4455, DNA_EASY_46X16250, DNA_EASY_5X1000,
    DNA_EASY_8X1252, DNA_MEDIUM_128X688,
};

fn run_for_sizes<Q: QMatrix + QMatrixMaker>(
    paths: &SequencePaths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let mut bench_group = criterion.benchmark_group(group_name);
    let mut bench = |id: &str, cost_fn: &mut PIPCost<Q>| {
        bench_group.bench_function(id, |bench| {
            bench.iter_custom(|iters| {
                let base_cost_fn = cost_fn.clone();

                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    let start = Instant::now();
                    let _ = black_box(cost_fn.cost());
                    elapsed += start.elapsed();
                    cost_fn.clone_from(&base_cost_fn);
                }
                elapsed
            });
        });
    };
    for (key, path) in paths {
        // safety: we immediately drop data and then the buffer after each bench
        // and never read from the pointer
        let mut data = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        bench(key, &mut data);
    }
    bench_group.finish();
}

fn pip_cost_dna_easy(criterion: &mut Criterion) {
    let paths = SequencePaths::from([
        ("5X1000", DNA_EASY_5X1000),
        ("8X1252", DNA_EASY_8X1252),
        ("17X2292", DNA_EASY_17X2292),
        ("33X4455", DNA_EASY_33X4455),
        ("46X16250", DNA_EASY_46X16250),
        ("128X688", DNA_MEDIUM_128X688),
    ]);
    run_for_sizes::<JC69>(&paths, "PIP Cost DNA", criterion);
}

fn pip_cost_aa_easy(criterion: &mut Criterion) {
    let paths = SequencePaths::from([
        ("6X97", AA_EASY_6X97),
        ("12X73", AA_EASY_12X73),
        ("14X165", AA_EASY_14X165),
        ("27X632", AA_EASY_27X632),
        ("45X223", AA_EASY_45X223),
        ("79X106", AA_MEDIUM_79X106),
    ]);
    run_for_sizes::<WAG>(&paths, "PIP Cost AA", criterion);
}

criterion_group! {
name = pip_cost_dna;
config = helpers::setup_suite();
targets = pip_cost_dna_easy,
}
criterion_group! {
name = pip_cost_aa;
config = helpers::setup_suite();
targets = pip_cost_aa_easy,
}
criterion_main!(pip_cost_dna, pip_cost_aa);
