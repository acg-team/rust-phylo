use std::fmt::Display;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use phylo::evolutionary_models::FrequencyOptimisation;
use phylo::likelihood::TreeSearchCost;
use phylo::optimisers::RegraftOptimiser;
use phylo::pip_model::PIPCost;
use phylo::substitution_models::{QMatrix, QMatrixMaker, GTR};

use phylo::{
    optimisers::ModelOptimiser,
    phylo_info::{PhyloInfo, PhyloInfoBuilder},
    pip_model::{PIPCostBuilder, PIPModel},
};

const ITERS: usize = 1000;
const DNA_EASY_17X2292: &str = "data/benchmark-datasets/dna/easy/wickd3a_7771.processed.fasta";

fn black_box_deterministic_phylo_info(seq_file: impl Into<PathBuf>) -> PhyloInfo {
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
fn black_box_pip_cost<Model: QMatrix + QMatrixMaker>(
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

fn find_best_regraft_for_single_spr_move<C: TreeSearchCost + Clone + Display + Send>(
    regraft_optimiser: &mut RegraftOptimiser<C>,
) -> anyhow::Result<f64> {
    let best_regraft = regraft_optimiser
        .find_max_cost_regraft_for_prune(f64::MIN)?
        .expect("invalid prune location for benchmarking");
    Ok(best_regraft.cost())
}

fn run_find_best_regraft_for_single_spr_move() {
    let cost_fn = black_box_pip_cost::<GTR>(DNA_EASY_17X2292, FrequencyOptimisation::Empirical);
    let tree = cost_fn.tree();
    // NOTE: regrafting an early preorder node would mean that a long path along the tree stays in tact
    // and less has to be re-calculated overall. We try to benchmark a likely worst case
    // since all parents have to be re-calculated
    let prune_location = *tree
        .postorder()
        .iter()
        .filter(|&n| n != &tree.root)
        .find(|prune| !cost_fn.tree().node(&tree.root).children.contains(prune))
        .expect("tree should have at least one node not a direct child of root");

    let mut elapsed = Duration::ZERO;
    for _ in 0..ITERS {
        let mut regraft_opt = RegraftOptimiser::new(&cost_fn, &prune_location);
        let start = Instant::now();
        let _ = black_box(find_best_regraft_for_single_spr_move(&mut regraft_opt));
        elapsed += start.elapsed();
    }
}

fn main() {
    run_find_best_regraft_for_single_spr_move();
}
