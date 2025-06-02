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
use phylo::pip_model::{PIPCost, PIPCostBuilder, PIPModel};
use phylo::substitution_models::{QMatrix, QMatrixMaker, WAG};
use phylo::tree::NodeIdx;

pub const AA_EASY_12X73: &str = "data/benchmark-datasets/aa/easy/nagya1_Cluster6386.aln";
pub const ITERS: usize = 20;

pub fn black_box_deterministic_phylo_info(seq_file: impl Into<PathBuf>) -> PhyloInfo {
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

fn single_spr_cycle<C: TreeSearchCost + Clone + Display + Send>(
    storage: &mut TopologyOptimiserStorage<C>,
    prune_locations: &[&NodeIdx],
) -> anyhow::Result<f64> {
    spr::fold_improving_moves(storage, f64::MIN, prune_locations)
}

fn run_single_spr_cycle_for_sizes<Q: QMatrix + QMatrixMaker + Send>(paths: &HashMap<&str, &str>) {
    let bench = |_id: &str,
                 (mut storage, prune_locations): (
        TopologyOptimiserStorage<PIPCost<Q>>,
        &[&NodeIdx],
    )| {
        let base_clone = storage.base_cost_fn().clone();
        let mut elapsed = Duration::ZERO;
        for _ in 0..ITERS {
            let start = Instant::now();
            let _ = black_box(single_spr_cycle(&mut storage, prune_locations));
            elapsed += start.elapsed();
            storage.set_base_cost_fn_to(&base_clone);
        }
        elapsed
    };
    for (key, path) in paths {
        let cost_fn = black_box_pip_cost::<Q>(path, FrequencyOptimisation::Empirical);
        let prune_locations = cost_fn
            .tree()
            .find_possible_prune_locations()
            .copied()
            .collect_vec();
        let prune_locations_ref = prune_locations.iter().collect_vec();
        let storage = TopologyOptimiserStorage::new_inplace(&cost_fn);
        bench(key, (storage, &prune_locations_ref));
    }
}

fn spr_aa() {
    let paths = HashMap::from([("12X73", AA_EASY_12X73)]);
    run_single_spr_cycle_for_sizes::<WAG>(&paths);
}

fn main() {
    spr_aa();
}
