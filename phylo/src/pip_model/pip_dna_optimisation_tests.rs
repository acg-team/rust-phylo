use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvolutionaryModel, EvolutionaryModelParameters,
    ModelType::DNA,
};
use crate::likelihood::LikelihoodCostFunction;
use crate::phylo_info::{GapHandling, PhyloInfo};
use crate::pip_model::pip_model_optimiser::PIPDNAModelOptimiser;
use crate::pip_model::{PIPDNAParams, PIPLikelihoodCost, PIPModel};

#[test]
fn check_parameter_optimisation_pip_arpiptest() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/pip/arpip/msa.fasta"),
        PathBuf::from("./data/pip/arpip/tree.nwk"),
        &GapHandling::Proper,
    )
    .unwrap();
    let likelihood = PIPLikelihoodCost { info: &info };
    let pip_params = PIPDNAParams::new(
        &DNAModelType::GTR,
        &[
            0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let model = PIPModel::new(DNA(GTR), &Vec::<f64>::from(pip_params.clone())).unwrap();
    let initial_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    let (_, _, logl) = PIPDNAModelOptimiser::new(&likelihood)
        .optimise_gtr_parameters(&pip_params)
        .unwrap();
    assert!(logl > initial_logl);
}

#[test]
fn test_optimisation_pip_propip_example() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/pip/propip/msa.initial.fasta"),
        PathBuf::from("./data/pip/propip/tree.nwk"),
        &GapHandling::Proper,
    )
    .unwrap();
    let likelihood = PIPLikelihoodCost { info: &info };
    let pip_params = PIPDNAParams::new(&DNAModelType::JC69, &[14.142_1, 0.1414, 1.0]).unwrap();
    let model = PIPModel::new(DNA(GTR), &Vec::<f64>::from(pip_params.clone())).unwrap();
    let initial_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert_relative_eq!(initial_logl, -1241.9944955187807, epsilon = 1e-3);
    let (_, optimised_params, logl) = PIPDNAModelOptimiser::new(&likelihood)
        .optimise_jc69_parameters(&pip_params)
        .unwrap();
    assert!(logl > initial_logl);
    assert_relative_eq!(logl, -1136.3884248861254, epsilon = 1e-5);
    let model = PIPModel::new(DNA(GTR), &Vec::<f64>::from(optimised_params.clone())).unwrap();
    let recomp_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    assert_eq!(logl, recomp_logl);
}

#[test]
fn check_example_against_python_no_gaps() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let cost = PIPLikelihoodCost::<4> { info: &info };
    let pip_params = PIPDNAParams::new(
        &DNAModelType::HKY,
        &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0],
    )
    .unwrap();
    let model = PIPModel::new(DNA(HKY), &Vec::<f64>::from(pip_params.clone())).unwrap();
    assert_relative_eq!(
        LikelihoodCostFunction::compute_log_likelihood(&cost, &model),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );
    let (_, opt_params, logl) = PIPDNAModelOptimiser::new(&cost)
        .optimise_hky_parameters(&pip_params)
        .unwrap();
    assert_eq!(opt_params.subst_params.rtc, opt_params.subst_params.rag);
    assert_eq!(opt_params.subst_params.rca, opt_params.subst_params.rta);
    assert_eq!(opt_params.subst_params.rca, opt_params.subst_params.rcg);
    assert_eq!(opt_params.subst_params.rca, opt_params.subst_params.rtg);
    assert_ne!(opt_params.mu, 1.2);
    assert_ne!(opt_params.lambda, 0.45);
    assert!(logl > -361.1613531649497);
    assert_relative_eq!(
        logl,
        -227.1894519082493, // value from the python script
        epsilon = 1e-1
    );
}

#[test]
fn check_parameter_optimisation_pip_gtr() {
    let info = PhyloInfo::from_files(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
        &GapHandling::Proper,
    )
    .unwrap();
    let likelihood = PIPLikelihoodCost { info: &info };
    let pip_params = PIPDNAParams::new(
        &DNAModelType::GTR,
        &[
            0.1, 0.1, 0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let model = PIPModel::new(DNA(GTR), &Vec::<f64>::from(pip_params.clone())).unwrap();
    let initial_logl = LikelihoodCostFunction::compute_log_likelihood(&likelihood, &model);
    let (_, opt_params, optimised_logl) = PIPDNAModelOptimiser::new(&likelihood)
        .optimise_gtr_parameters(&pip_params)
        .unwrap();
    assert_relative_eq!(initial_logl, -9988.486546494, epsilon = 1e0); // value from the python script
    assert!(optimised_logl > initial_logl);
    // comparing to optimised parameter values from check_parameter_optimisation_gtr
    assert_relative_eq!(opt_params.subst_params.rtc, 1.03398, epsilon = 1e-4);
    assert_relative_eq!(opt_params.subst_params.rta, 0.03189, epsilon = 1e-4);
    assert_relative_eq!(opt_params.subst_params.rtg, 0.00001, epsilon = 1e-4);
    assert_relative_eq!(opt_params.subst_params.rca, 0.07906, epsilon = 1e-4);
    assert_relative_eq!(opt_params.subst_params.rcg, 0.04276, epsilon = 1e-4);
    assert_relative_eq!(opt_params.subst_params.rag, 1.00000, epsilon = 1e-4);
}
