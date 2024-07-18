use std::path::PathBuf;

use crate::evolutionary_models::{
    DNAModelType::*, EvolutionaryModelParameters, FrequencyOptimisation,
};
use crate::optimisers::dna_model_optimiser::DNAModelOptimiser;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::substitution_models::dna_models::{DNALikelihoodCost, DNASubstParams};

#[test]
fn branch_length_opt_gtr() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };

    let params =
        DNASubstParams::new(&GTR, &[0.25, 0.35, 0.3, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

    let (_, optim_params, _) = BranchLength::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(optim_params.pi.as_slice() == [0.25, 0.35, 0.3, 0.1]);
}
