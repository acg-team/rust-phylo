use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::DNAModelType::*;
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{FrequencyOptimisation, ModelOptimiser, PIPParamOptimiser};
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::{PIPCost, PIPModel};
use crate::substitution_models::dna_models::DNASubstModel;

#[test]
fn check_parameter_optimisation_pip_arpiptest() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/pip/arpip/msa.fasta"),
        PathBuf::from("./data/pip/arpip/tree.nwk"),
    )
    .build()
    .unwrap();

    let pip_gtr = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let llik = PIPCost { model: &pip_gtr };
    let o = PIPParamOptimiser::new(&llik, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let initial_logl = llik.cost(info);
    assert_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
}

#[test]
fn test_optimisation_pip_propip_example() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/pip/propip/msa.initial.fasta"),
        PathBuf::from("./data/pip/propip/tree.nwk"),
    )
    .build()
    .unwrap();

    let pip_gtr = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            14.142_1, 0.1414, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let likelihood = PIPCost { model: &pip_gtr };
    let initial_logl = likelihood.cost(info);
    assert_relative_eq!(initial_logl, -1241.9944955187807, epsilon = 1e-3);
    let o = PIPParamOptimiser::new(&likelihood, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
    assert!(o.final_logl > -1136.3884248861254);
    let pip_optimised = PIPModel::create(&o.model.params);
    let optimised_llik = PIPCost {
        model: &pip_optimised,
    };
    assert_eq!(o.final_logl, optimised_llik.cost(info));
}

#[test]
fn check_example_against_python_no_gaps() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_hky =
        PIPModel::<DNASubstModel>::new(HKY, &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0]).unwrap();
    let llik = PIPCost { model: &pip_hky };
    assert_relative_eq!(
        llik.cost(info),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );
    let o = PIPParamOptimiser::new(&llik, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let params = o.model.params;
    assert_eq!(params.subst_params.rtc, params.subst_params.rag);
    assert_eq!(params.subst_params.rca, params.subst_params.rta);
    assert_eq!(params.subst_params.rca, params.subst_params.rcg);
    assert_eq!(params.subst_params.rca, params.subst_params.rtg);
    assert_ne!(params.mu, 1.2);
    assert_ne!(params.lambda, 0.45);
    assert!(o.final_logl > -361.1613531649497);
    assert_relative_eq!(
        o.final_logl,
        -227.1894519082493, // value from the python script
        epsilon = 1e-1
    );
}

#[test]
fn check_parameter_optimisation_pip_gtr() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let pip_gtr = PIPModel::<DNASubstModel>::new(
        GTR,
        &[
            0.1, 0.1, 0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let llik = PIPCost { model: &pip_gtr };
    let initial_logl = llik.cost(info);
    let o = PIPParamOptimiser::new(&llik, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    assert_relative_eq!(initial_logl, -9988.486546494, epsilon = 1e0); // value from the python script
    assert_relative_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
    let subst_params = o.model.params.subst_params;
    // comparing to optimised parameter values from check_parameter_optimisation_gtr
    assert_relative_eq!(subst_params.rtc, 1.03398, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rta, 0.03189, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rtg, 0.00001, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rca, 0.07906, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rcg, 0.04276, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rag, 1.00000, epsilon = 1e-4);
}
