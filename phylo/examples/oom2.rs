use std::collections::HashMap;
use std::fmt::Display;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use itertools::Itertools;
use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::{spr, ModelOptimiser, TopologyOptimiserStorage};
use phylo::phylo_info::{PhyloInfo, PhyloInfoBuilder};
use phylo::pip_model::{
    PIPCost, PIPCostBuilder, PIPModel, PIPModelCacheBuf, PIPModelCacheBufDimensions,
};
use phylo::substitution_models::{QMatrix, QMatrixMaker, GTR, WAG};
use phylo::tree::NodeIdx;

pub const AA_EASY_12X73: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster6386.aln";

pub fn black_box_deterministic_phylo_info(seq_file: impl Into<PathBuf>) -> PhyloInfo {
    black_box(
        PhyloInfoBuilder::new(seq_file.into())
            .build()
            .expect("sequence file should be able to build phylo info"),
    )
}

pub fn black_box_pip_cost<Model: QMatrix + QMatrixMaker>(
    path: impl Into<PathBuf>,
) -> PIPCost<Model> {
    let info = black_box_deterministic_phylo_info(path);
    let pip_cost = PIPCostBuilder::new(PIPModel::<Model>::new(&[], &[]), info)
        .build()
        .expect("failed to build pip cost optimiser");

    // done for a more 'realistic' setup
    black_box(pip_cost)
}

fn clone_cost_fns() {
    let cost_fn = black_box_pip_cost::<WAG>(AA_EASY_12X73);
    black_box(vec![cost_fn; 4_000]);
}
fn clone_caches() {
    let dimensions = PIPModelCacheBufDimensions::new(21, 73, 2 * 12 - 1);
    let owned_cache = black_box(PIPModelCacheBuf::new_owned(dimensions));
    black_box(vec![owned_cache; 1_000]);
}
fn clone_boxes() {
    let dimensions = PIPModelCacheBufDimensions::new(21, 73, 2 * 12 - 1);
    let owned_cache = black_box(PIPModelCacheBuf::new_owned(dimensions));
    black_box(vec![owned_cache; 1_000]);
}

pub const ITERS: usize = 10_000;
fn main() {
    for i in 0..ITERS {
        clone_cost_fns();
        // clone_caches();
        if i % 1000 == 0 {
            println!("finished {i:0>8}");
        }
    }
}
