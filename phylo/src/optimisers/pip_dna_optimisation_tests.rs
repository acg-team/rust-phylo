use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{
    DNAModelType::{self, *},
    EvolutionaryModel,
};
use crate::likelihood::LikelihoodCostFunction;
use crate::optimisers::pip_model_optimiser::PIPDNAModelOptimiser;
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::{PIPDNAParams, PIPLikelihoodCost, PIPModel};

#[test]
fn check_parameter_optimisation_pip_arpiptest() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/pip/arpip/msa.fasta"),
        PathBuf::from("./data/pip/arpip/tree.nwk"),
    )
    .build()
    .unwrap();

    let model = PIPModel::new(
        GTR,
        &[
            0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let likelihood = PIPLikelihoodCost {
        info,
        model: &model,
    };

    let initial_logl = LikelihoodCostFunction::compute_logl(&likelihood);
    let (_, _, logl) = PIPDNAModelOptimiser::new(&likelihood)
        .optimise_parameters()
        .unwrap();
    assert!(logl > initial_logl);
}

#[test]
fn test_optimisation_pip_propip_example() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/pip/propip/msa.initial.fasta"),
        PathBuf::from("./data/pip/propip/tree.nwk"),
    )
    .build()
    .unwrap();

    let model = PIPModel::new(
        DNAModelType::GTR,
        &[
            14.142_1, 0.1414, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let likelihood = PIPLikelihoodCost {
        info: info.clone(),
        model: &model,
    };
    let initial_logl = LikelihoodCostFunction::compute_logl(&likelihood);
    assert_relative_eq!(initial_logl, -1241.9944955187807, epsilon = 1e-3);
    let (_, optimised_params, logl) = PIPDNAModelOptimiser::new(&likelihood)
        .optimise_parameters()
        .unwrap();
    assert!(logl > initial_logl);
    assert!(logl > -1136.3884248861254);
    let model = PIPModel::create(&optimised_params);

    let likelihood = PIPLikelihoodCost {
        info,
        model: &model,
    };
    let recomp_logl = LikelihoodCostFunction::compute_logl(&likelihood);
    assert_eq!(logl, recomp_logl);
}

#[test]
fn check_example_against_python_no_gaps() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_params = PIPDNAParams::new(
        &DNAModelType::HKY,
        &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0],
    )
    .unwrap();
    let model = PIPModel::create(&pip_params);
    let cost = PIPLikelihoodCost {
        info,
        model: &model,
    };

    assert_relative_eq!(
        LikelihoodCostFunction::compute_logl(&cost),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );
    let (_, opt_params, logl) = PIPDNAModelOptimiser::new(&cost)
        .optimise_parameters()
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
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();

    let pip_params = PIPDNAParams::new(
        &DNAModelType::GTR,
        &[
            0.1, 0.1, 0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let model = PIPModel::create(&pip_params);
    let likelihood = PIPLikelihoodCost {
        info,
        model: &model,
    };
    let initial_logl = LikelihoodCostFunction::compute_logl(&likelihood);
    let (_, opt_params, optimised_logl) = PIPDNAModelOptimiser::new(&likelihood)
        .optimise_parameters()
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
