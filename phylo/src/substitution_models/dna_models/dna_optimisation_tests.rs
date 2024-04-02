use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::EvolutionaryModel;
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{phyloinfo_from_files, GapHandling};
use crate::substitution_models::dna_models::{
    dna_model_optimiser::DNAModelOptimiser,
    gtr::{self},
    parse_k80_parameters, DNALikelihoodCost, DNASubstModel, DNASubstParams,
};
use crate::substitution_models::FreqVector;

#[test]
fn check_likelihood_opt_k80() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();

    let likelihood = DNALikelihoodCost { info: &info };
    let model = DNASubstModel::new("k80", &[4.0, 1.0]).unwrap();
    let params = parse_k80_parameters(&[4.0, 1.0]).unwrap();
    let unopt_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);

    let (_, _, logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_k80_parameters(&params)
        .unwrap();
    assert!(logl > unopt_logl);

    let model = DNASubstModel::new("k80", &[1.884815, 1.0]).unwrap();
    let expected_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);

    assert_relative_eq!(logl, expected_logl, epsilon = 1e-6);
    assert_relative_eq!(logl, -4034.5008033, epsilon = 1e-6);
}

#[test]
fn check_parameter_optimisation_gtr() {
    let info = phyloinfo_from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Ambiguous,
    )
    .unwrap();
    let likelihood = DNALikelihoodCost { info: &info };
    let phyml_params = DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.24720, 0.35320, 0.29540, 0.10420]),
        rtc: 1.0,
        rta: 0.031184397,
        rtg: 0.000100000,
        rca: 0.077275972,
        rcg: 0.041508690,
        rag: 1.0,
    }; // Optimized parameters from PhyML
    let model = gtr::gtr(phyml_params);
    let phyml_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_params = DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.25318, 0.32894, 0.31196, 0.10592]),
        rtc: 0.88892,
        rta: 0.03190,
        rtg: 0.00001,
        rca: 0.07102,
        rcg: 0.02418,
        rag: 1.0,
    }; // Original input to paml
    let model = gtr::gtr(paml_params);
    let paml_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert!(phyml_logl > paml_logl);

    let params = DNASubstParams {
        pi: FreqVector::from_column_slice(&[0.24720, 0.35320, 0.29540, 0.10420]),
        rtc: 1.0,
        rta: 1.0,
        rtg: 1.0,
        rca: 1.0,
        rcg: 1.0,
        rag: 1.0,
    };
    let (_, opt_params, logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_gtr_parameters(&params)
        .unwrap();
    assert!(logl > phyml_logl);
    assert!(logl > paml_logl);

    let (iters, _, double_opt_logl) = DNAModelOptimiser::new(&likelihood)
        .optimise_gtr_parameters(&opt_params)
        .unwrap();
    assert!(double_opt_logl >= logl);
    assert!(iters < 10);
}
