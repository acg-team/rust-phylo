use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use phylo::{
    likelihood::ModelSearchCost,
    phylo_info::PhyloInfoBuilder,
    substitution_models::{SubstModel, SubstitutionCostBuilder, JC69},
};

fn pip_cost_tiny_but_long(criterion: &mut Criterion) {
    let info = black_box(
        PhyloInfoBuilder::new("data/sim/GTR/gtr.fasta".into())
            .build()
            .unwrap(),
    );
    let jc69 = black_box(SubstModel::<JC69>::new(&[], &[]));
    let c = black_box(SubstitutionCostBuilder::new(jc69, info).build().unwrap());
    criterion.bench_function("PIP cost tiny but long", |bench| {
        bench.iter(|| {
            // clone because of interior mutability
            c.clone().cost();
        });
    });
}

criterion_group! {
name = pip_cost;
config = Criterion::default();
targets = pip_cost_tiny_but_long
}
criterion_main!(pip_cost);
