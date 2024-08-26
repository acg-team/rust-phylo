use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{DNAModelType::*, EvoModel, ProteinModelType::*};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::{EvoModelOptimiser, FrequencyOptimisation, ModelOptimiser};
use crate::phylo_info::PhyloInfoBuilder;
use crate::pip_model::PIPModel;
use crate::substitution_models::{DNASubstModel, ProteinSubstModel};

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
    let o = ModelOptimiser::new(&pip_gtr, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let initial_logl = pip_gtr.cost(info);
    assert_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
}

#[test]
fn optimisation_pip_propip_example() {
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

    let initial_logl = pip_gtr.cost(info);
    assert_relative_eq!(initial_logl, -1241.9555557710014, epsilon = 1e-1);
    let o = ModelOptimiser::new(&pip_gtr, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);
    assert_relative_eq!(o.final_logl, -1081.1682773217494, epsilon = 1e-0);
    assert_eq!(o.final_logl, o.model.cost(info));
}

#[test]
fn optimisation_against_python_no_gaps() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/Huelsenbeck_example_long_DNA.fasta"),
        PathBuf::from("./data/Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_hky =
        PIPModel::<DNASubstModel>::new(HKY, &[1.2, 0.45, 0.25, 0.25, 0.25, 0.25, 1.0]).unwrap();
    assert_relative_eq!(
        pip_hky.cost(info),
        -361.1613531649497, // value from the python script
        epsilon = 1e-1
    );
    let o = ModelOptimiser::new(&pip_hky, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let params = &o.model.params.subst_model.params;
    assert_eq!(params.rtc, params.rag);
    assert_eq!(params.rca, params.rta);
    assert_eq!(params.rca, params.rcg);
    assert_eq!(params.rca, params.rtg);
    assert_ne!(o.model.params.mu, 1.2);
    assert_ne!(o.model.params.lambda, 0.45);
    assert!(o.final_logl > -361.1613531649497);
    assert_relative_eq!(
        o.final_logl,
        -227.1894519082493, // value from the python script
        epsilon = 1e-1
    );
}

#[test]
fn optimisation_pip_gtr() {
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
    let initial_logl = pip_gtr.cost(info);
    let o = ModelOptimiser::new(&pip_gtr, info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert_relative_eq!(initial_logl, -9988.486546494, epsilon = 1e0); // value from the python script
    assert_relative_eq!(o.initial_logl, initial_logl);
    assert!(o.final_logl > initial_logl);

    let subst_params = o.model.params.subst_model.params;
    // comparing to optimised parameter values from check_parameter_optimisation_gtr
    assert_relative_eq!(subst_params.rtc, 1.03398, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rta, 0.03189, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rtg, 0.00001, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rca, 0.07906, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rcg, 0.04276, epsilon = 1e-4);
    assert_relative_eq!(subst_params.rag, 1.00000, epsilon = 1e-4);
}

#[test]
fn protein_example_pip_opt() {
    let info = &PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/phyml_protein_nogap_example.fasta"),
        PathBuf::from("./data/phyml_protein_example.newick"),
    )
    .build()
    .unwrap();
    let pip = PIPModel::<ProteinSubstModel>::new(WAG, &[2.0, 0.1]).unwrap();
    let initial_logl = pip.cost(info);
    let o = ModelOptimiser::new(&pip, info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.final_logl > initial_logl);
    assert_relative_eq!(o.initial_logl, initial_logl);
    assert_ne!(o.model.params.lambda, 2.0);
    assert_ne!(o.model.params.mu, 0.1);
    assert_eq!(o.model.cost(info), o.final_logl);
}
