use std::path::PathBuf;

use crate::evolutionary_models::DNAModelType;
use crate::evolutionary_models::EvolutionaryModel;
use crate::optimisers::branch_length_optimiser::BranchOptimiser;
use crate::phylo_info::{GapHandling, PhyloInfoBuilder};
use crate::pip_model::PIPDNAModel;

#[test]
fn branch_optimiser_likelihood_increase() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        GapHandling::Proper,
    )
    .build()
    .unwrap();
    let model = PIPDNAModel::new(
        DNAModelType::GTR,
        &[
            14.142_1, 0.1414, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let optimiser = BranchOptimiser::new(&model, &info);
    let (_iters, _tree, init_logl, opt_logl) = optimiser.optimise_parameters().unwrap();
    assert!(opt_logl > init_logl);
}
