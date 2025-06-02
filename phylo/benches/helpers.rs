#![allow(dead_code)]
use core::slice;
/// this file is essentially a workaround for #[cfg(test)] like behaviour for the benchmarks
/// The dev-depencies are only available in benchmarks or tests
use std::{collections::HashMap, hint::black_box, mem::MaybeUninit, path::PathBuf, time::Duration};

use criterion::Criterion;
use phylo::{
    evolutionary_models::FrequencyOptimisation,
    optimisers::ModelOptimiser,
    phylo_info::{PhyloInfo, PhyloInfoBuilder},
    pip_model::{PIPCost, PIPCostBuilder, PIPModel},
    substitution_models::{QMatrix, QMatrixMaker},
    util::mem::boxed::BoxSlice,
};

pub type BenchPath = &'static str;
pub type SequencePaths = HashMap<&'static str, BenchPath>;

pub const N10_M1000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m1000.fasta";
pub const N10_M5000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m5000.fasta";
pub const N10_M10000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m10000.fasta";
pub const N10_M20000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m20000.fasta";
pub const N10_M50000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m50000.fasta";
pub const N10_M100000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m100000.fasta";
pub const N10_M200000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m200000.fasta";
pub const N10_M300000: &str = "data/benchmark-datasets/one_shot/seq_length/n10m300000.fasta";

pub const N10_M500: &str = "data/benchmark-datasets/one_shot/taxa/n10m500.fasta";
pub const N20_M500: &str = "data/benchmark-datasets/one_shot/taxa/n20m500.fasta";
pub const N30_M500: &str = "data/benchmark-datasets/one_shot/taxa/n30m500.fasta";
pub const N40_M500: &str = "data/benchmark-datasets/one_shot/taxa/n40m500.fasta";
pub const N50_M500: &str = "data/benchmark-datasets/one_shot/taxa/n50m500.fasta";
pub const N60_M500: &str = "data/benchmark-datasets/one_shot/taxa/n60m500.fasta";
pub const N70_M500: &str = "data/benchmark-datasets/one_shot/taxa/n70m500.fasta";
pub const N80_M500: &str = "data/benchmark-datasets/one_shot/taxa/n80m500.fasta";
pub const N90_M500: &str = "data/benchmark-datasets/one_shot/taxa/n90m500.fasta";
pub const N100_M500: &str = "data/benchmark-datasets/one_shot/taxa/n100m500.fasta";
pub const N200_M500: &str = "data/benchmark-datasets/one_shot/taxa/n200m500.fasta";
pub const HIV_1: &str = "data/real_examples/HIV-1_env_DNA_mafft_alignment_subset.fasta";
pub const DNA_EASY_5X1000: &str = "data/sim/GTR/gtr.fasta";
pub const DNA_EASY_8X1252: &str = "data/benchmark-datasets/dna/easy/wickd3b_7705.processed.fasta";
pub const DNA_EASY_17X2292: &str = "data/benchmark-datasets/dna/easy/wickd3a_7771.processed.fasta";
pub const DNA_EASY_33X4455: &str =
    "data/benchmark-datasets/dna/easy/tarvd7_ENSG00000126777.nuc.ord.processed.fasta";
pub const DNA_EASY_46X16250: &str =
    "data/benchmark-datasets/dna/easy/jarvd5a_intron_1521.processed.fasta";

pub const DNA_MEDIUM_128X688: &str =
    "data/benchmark-datasets/dna/medium/misod2b_EOG58W9HG.processed.fasta";

pub const AA_EASY_6X97: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster9992.aln";
pub const AA_EASY_12X73: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster6386.aln";
pub const AA_EASY_12X445: &str = "data/benchmark-datasets/aa/easy/wicka3_7004.aln";
pub const AA_EASY_14X165: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster5493.aln";
pub const AA_EASY_45X223: &str = "data/benchmark-datasets/aa/easy/whela7_Gene_0917.aln";
pub const AA_EASY_27X632: &str = "data/benchmark-datasets/aa/easy/boroa6_OG126_gene166.fasta";

pub const AA_MEDIUM_79X106: &str = "data/benchmark-datasets/aa/medium/strua5_gene1339_23221.aln";
pub const AA_MEDIUM_30X86: &str = "data/benchmark-datasets/aa/medium/nagya1_Cluster3439.aln";

pub fn black_box_deterministic_phylo_info(seq_file: impl Into<PathBuf>) -> PhyloInfo {
    assert!(
        cfg!(feature = "deterministic"),
        "only run benches with '-F deterministic'"
    );
    black_box(
        PhyloInfoBuilder::new(seq_file.into())
            .build()
            .expect("sequence file should be able to build phylo info"),
    )
}

pub fn black_box_pip_cost<Model: QMatrix + QMatrixMaker>(
    path: impl Into<PathBuf>,
    freq_opt: FrequencyOptimisation,
) -> PIPCost<Model> {
    let info = black_box_deterministic_phylo_info(path);
    let pip_cost = PIPCostBuilder::new(PIPModel::<Model>::new(&[], &[]), info)
        .build()
        .expect("failed to build pip cost optimiser");

    // done for a more 'realistic' setup
    let model_optimiser = ModelOptimiser::new(pip_cost, freq_opt);
    black_box(
        model_optimiser
            .run()
            .expect("model optimiser should pass")
            .cost,
    )
}

/// # Safety
/// - As long as PIPCost doesnt have a lifetime be sure not to leak it above
///   the scope of the buffer and manually drop the buffer
/// - DO NOT READ FROM THE RETURNED POINTER, a mutable reference to it exists
///   in PIPCost making it UB
pub unsafe fn black_box_pip_cost_with_cache_buf<Model: QMatrix + QMatrixMaker>(
    path: impl Into<PathBuf>,
    freq_opt: FrequencyOptimisation,
) -> (PIPCost<Model>, *mut [MaybeUninit<f64>]) {
    let pip_cost_in_regular_buf = black_box_pip_cost(path, freq_opt);

    let cache_buf = BoxSlice::alloc_slice_uninit(
        pip_cost_in_regular_buf
            .cache_dimensions()
            .total_len_f64_padded()
            .try_into()
            .unwrap(),
    );
    let leaked_buf: *mut [MaybeUninit<f64>] = cache_buf.leak();
    let leaked_buf_len = leaked_buf.len();
    let leaked_buf_ptr = unsafe { leaked_buf.as_mut() }.unwrap().as_mut_ptr();

    // safety: lifetime of the pipcost is tied to this buf, as noted in doc comments
    let cost_in_cache_buf = pip_cost_in_regular_buf
        .clone_with_cache_in(unsafe { slice::from_raw_parts_mut(leaked_buf_ptr, leaked_buf_len) });
    (cost_in_cache_buf, leaked_buf)
}

#[derive(Clone)]
pub struct PIPConfig {
    pub freqs: Vec<f64>,
    pub params: Vec<f64>,
    pub freq_opt: FrequencyOptimisation,
    pub max_iters: usize,
    pub epsilon: f64,
}
pub fn black_box_raw_pip_cost_with_config<Model: QMatrix + QMatrixMaker>(
    seq_path: impl Into<PathBuf>,
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
        PIPModel::<Model>::new(&cfg.freqs, &cfg.params),
        info,
    ))
    .build()
    .expect("failed to build pip cost optimiser");

    (cfg, pip_cost)
}

pub fn setup_suite() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(60))
        .with_profiler(pprof::criterion::PProfProfiler::new(
            997, // taken from the flamegraph repo's default
            pprof::criterion::Output::Flamegraph(None),
        ))
}

/// empty on purpose, there are no benches here but the crate still needs
/// to be runnable otherwise criterion crashes
fn main() {}
