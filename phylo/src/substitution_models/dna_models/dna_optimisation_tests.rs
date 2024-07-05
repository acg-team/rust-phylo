use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{
    DNAModelType::*, EvolutionaryModel, EvolutionaryModelParameters, FrequencyOptimisation,
    ModelType::DNA,
};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::substitution_models::dna_models::{
    dna_model_optimiser::DNAModelOptimiser, make_dna_model, DNALikelihoodCost, DNASubstModel,
    DNASubstParams,
};

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
        .optimise_parameters(&params, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(logl > unopt_logl);

    let model = DNASubstModel::new(DNA(K80), &[1.884815, 1.0]).unwrap();
    let expected_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);

    assert_relative_eq!(logl, expected_logl, epsilon = 1e-6);
    assert_relative_eq!(logl, -4034.5008033, epsilon = 1e-6);
}

#[test]
fn frequencies_unchanged_opt_k80() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let params = DNASubstParams::new(&K80, &[4.0, 1.0]).unwrap();

    let (_, optim_params, _) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Empirical)
        .unwrap();
    assert!(optim_params.pi.iter().all(|&x| x == 0.25));
}

#[test]
fn parameter_definition_after_optim_k80() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let params = DNASubstParams::new(&K80, &[4.0, 1.0]).unwrap();

    let (_, optim_parameters, _) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Fixed)
        .unwrap();
    assert_eq!(optim_parameters.rtc, optim_parameters.rag);
    assert_eq!(optim_parameters.rta, optim_parameters.rtg);
    assert_eq!(optim_parameters.rta, optim_parameters.rca);
    assert_eq!(optim_parameters.rta, optim_parameters.rcg);
    assert!(optim_parameters.pi.iter().all(|&x| x == 0.25));
}

#[test]
fn gtr_on_k80_data() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let params = DNASubstParams::new(
        &GTR,
        &[0.25, 0.35, 0.3, 0.1, 0.88, 0.03, 0.00001, 0.07, 0.02, 1.0],
    )
    .unwrap();

    let (_, optim_parameters, _) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Empirical)
        .unwrap();
    assert_relative_eq!(optim_parameters.rta, optim_parameters.rtg, epsilon = 1e-1);
    assert_relative_eq!(optim_parameters.rta, optim_parameters.rca, epsilon = 1e-1);
    assert_relative_eq!(optim_parameters.rta, optim_parameters.rcg, epsilon = 1e-1);
    assert!(optim_parameters.pi.iter().all(|&x| x - 0.25 < 1e-2));
}

#[test]
fn parameter_definition_after_optim_hky() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let params = DNASubstParams::new(&TN93, &[0.26, 0.2, 0.4, 0.14, 4.0, 1.0]).unwrap();

    let (_, optim_parameters, _) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Empirical)
        .unwrap();
    assert_eq!(optim_parameters.rtc, optim_parameters.rag);
    assert_eq!(optim_parameters.rta, optim_parameters.rtg);
    assert_eq!(optim_parameters.rta, optim_parameters.rca);
    assert_eq!(optim_parameters.rta, optim_parameters.rcg);
    assert!(optim_parameters.pi.iter().all(|&x| x != 0.25));
}

#[test]
fn parameter_definition_after_optim_tn93() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let params = DNASubstParams::new(&TN93, &[0.26, 0.2, 0.4, 0.14, 4.0, 2.0, 1.0]).unwrap();

    let (_, optim_parameters, _) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Empirical)
        .unwrap();
    assert_ne!(optim_parameters.rtc, optim_parameters.rag);
    assert_eq!(optim_parameters.rta, optim_parameters.rtg);
    assert_eq!(optim_parameters.rta, optim_parameters.rca);
    assert_eq!(optim_parameters.rta, optim_parameters.rcg);
    assert!(optim_parameters.pi.iter().all(|&x| x != 0.25));
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
        .optimise_parameters(&params, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(logl > phyml_logl);
    assert!(logl > paml_logl);

    let (iters, _, double_opt_logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(double_opt_logl >= logl);
    assert!(iters < 10);
}

#[test]
fn frequencies_fixed_opt_gtr() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };

    let params =
        DNASubstParams::new(&GTR, &[0.25, 0.35, 0.3, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

    let (_, optim_params, _) = DNAModelOptimiser::new(&likelihood)
        .optimise_parameters(&params, FrequencyOptimisation::Fixed)
        .unwrap();
    assert!(optim_params.pi.as_slice() == [0.25, 0.35, 0.3, 0.1]);
}
