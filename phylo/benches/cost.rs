use std::{hint::black_box, path::PathBuf};

use criterion::{criterion_group, criterion_main, Criterion};
use phylo::{
    likelihood::ModelSearchCost,
    phylo_info::PhyloInfoBuilder,
    pip_model::{PIPCost, PIPCostBuilder, PIPModel},
    substitution_models::{QMatrix, QMatrixMaker, JC69, WAG},
};

fn setup<Q: QMatrix + QMatrixMaker>(seq_path: impl Into<PathBuf>) -> PIPCost<Q> {
    let info = black_box(PhyloInfoBuilder::new(seq_path.into()).build().unwrap());
    let pip_model = black_box(PIPModel::<Q>::new(&[], &[]));
    black_box(PIPCostBuilder::new(pip_model, info).build().unwrap())
}

struct Paths {
    small: &'static str,
    medium: &'static str,
    large: &'static str,
}
fn run_for_sizes<Q: QMatrix + QMatrixMaker>(
    paths: Paths,
    group_name: &'static str,
    criterion: &mut Criterion,
) {
    let small = setup::<Q>(paths.small);
    let medium = setup::<Q>(paths.medium);
    let large = setup::<Q>(paths.large);
    let mut dna_easy = criterion.benchmark_group(group_name);
    let mut bench = |id: &str, data: PIPCost<Q>| {
        dna_easy.bench_function(id, |bench| {
            bench.iter_batched(
                // clone because of interior mutability
                || data.clone(),
                |cost_fn| cost_fn.cost(),
                criterion::BatchSize::SmallInput,
            );
        });
    };
    bench("small", small);
    bench("medium", medium);
    bench("large", large);
}

fn pip_cost_dna_easy(criterion: &mut Criterion) {
    let paths = Paths {
        small: "data/benchmark-datasets/dna/easy/wickd3b_7705.processed.fasta",
        medium: "data/benchmark-datasets/dna/easy/tarvd7_ENSG00000126777.nuc.ord.processed.fasta",
        large: "data/benchmark-datasets/dna/easy/misod2a_EOG5CVDNS.processed.fasta",
    };
    run_for_sizes::<JC69>(paths, "DNA easy", criterion);
}

fn pip_cost_aa_easy(criterion: &mut Criterion) {
    let paths = Paths {
        small: "data/benchmark-datasets/aa/easy/nagya1_Cluster5493.aln",
        medium: "data/benchmark-datasets/aa/easy/strua5_gene1642_23524.aln",
        large: "data/benchmark-datasets/aa/easy/yanga8_cc3720-1.inclade1.ortho1.aln",
    };
    run_for_sizes::<WAG>(paths, "AA easy", criterion);
}

criterion_group! {
name = pip_cost_dna;
config = Criterion::default();
targets = pip_cost_dna_easy,
}
criterion_group! {
name = pip_cost_aa;
config = Criterion::default();
targets = pip_cost_aa_easy,
}
criterion_main!(pip_cost_dna, pip_cost_aa);
