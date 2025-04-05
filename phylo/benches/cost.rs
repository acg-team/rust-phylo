use std::{hint::black_box, path::PathBuf};

use criterion::{criterion_group, criterion_main, Criterion};
use phylo::{
    likelihood::ModelSearchCost,
    phylo_info::PhyloInfoBuilder,
    pip_model::{PIPCost, PIPCostBuilder, PIPModel},
    substitution_models::{QMatrix, QMatrixMaker, JC69},
};

fn setup<Q: QMatrix + QMatrixMaker>(seq_path: impl Into<PathBuf>) -> PIPCost<Q> {
    let info = black_box(PhyloInfoBuilder::new(seq_path.into()).build().unwrap());
    let jc69 = black_box(PIPModel::<Q>::new(&[], &[]));
    black_box(PIPCostBuilder::new(jc69, info).build().unwrap())
}

fn pip_cost_dna_easy_small(criterion: &mut Criterion) {
    let c = setup::<JC69>("data/benchmark-datasets/dna/easy/wickd3b_7705.processed.fasta");
    criterion.bench_function("DNA easy small", |bench| {
        bench.iter_batched(
            // clone because of interior mutability
            || c.clone(),
            |cost_fn| cost_fn.cost(),
            criterion::BatchSize::SmallInput,
        );
    });
}

fn pip_cost_dna_easy_medium(criterion: &mut Criterion) {
    let c = setup::<JC69>(
        "data/benchmark-datasets/dna/easy/tarvd7_ENSG00000126777.nuc.ord.processed.fasta",
    );
    criterion.bench_function("DNA easy medium", |bench| {
        bench.iter_batched(
            // clone because of interior mutability
            || c.clone(),
            |cost_fn| cost_fn.cost(),
            criterion::BatchSize::SmallInput,
        );
    });
}

fn pip_cost_dna_easy_large(criterion: &mut Criterion) {
    let c = setup::<JC69>("data/benchmark-datasets/dna/easy/misod2a_EOG5CVDNS.processed.fasta");
    criterion.bench_function("DNA easy large", |bench| {
        bench.iter_batched(
            // clone because of interior mutability
            || c.clone(),
            |cost_fn| cost_fn.cost(),
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group! {
name = pip_cost_easy;
config = Criterion::default();
targets = pip_cost_dna_easy_small, pip_cost_dna_easy_medium, pip_cost_dna_easy_large,
}
criterion_main!(pip_cost_easy);
