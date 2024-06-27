use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{
    EvolutionaryModel, EvolutionaryModelParameters, FrequencyOptimisation,
};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::substitution_models::dna_models::{
    dna_model_optimiser::DNAModelOptimiser, make_dna_model, DNALikelihoodCost, DNAModelType::*,
    DNASubstModel, DNASubstParams,
};
use crate::substitution_models::ModelType::DNA;

#[test]
fn check_likelihood_opt_k80() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let model = DNASubstModel::new(DNA(JC69), &[4.0, 1.0]).unwrap();

    let params = DNASubstParams::new(&K80, &[4.0, 1.0]).unwrap();
    let unopt_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);

    let (_, _, logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, K80, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(logl > unopt_logl);

    let model = DNASubstModel::new(DNA(K80), &[1.884815, 1.0]).unwrap();
    let expected_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);

    assert_relative_eq!(logl, expected_logl, epsilon = 1e-6);
    assert_relative_eq!(logl, -4034.5008033, epsilon = 1e-6);
}

#[test]
fn check_parameter_optimisation_gtr() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let phyml_params = DNASubstParams::new(
        &GTR,
        &[
            0.24720,
            0.35320,
            0.29540,
            0.10420,
            1.0,
            0.031184397,
            0.000100000,
            0.077275972,
            0.041508690,
            1.0,
        ],
    )
    .unwrap();
    // Optimized parameters from PhyML

    let model = make_dna_model(phyml_params);
    let phyml_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_params = DNASubstParams::new(
        &GTR,
        &[
            0.25318, 0.32894, 0.31196, 0.10592, 0.88892, 0.03190, 0.00001, 0.07102, 0.02418, 1.0,
        ],
    )
    .unwrap(); // Original input to paml
    let model = make_dna_model(paml_params);
    let paml_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert!(phyml_logl > paml_logl);

    let params = DNASubstParams::new(
        &GTR,
        &[
            0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let (_, _, logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, GTR, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(logl > phyml_logl);
    assert!(logl > paml_logl);

    let (iters, _, double_opt_logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, GTR, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(double_opt_logl >= logl);
    assert!(iters < 10);
}
