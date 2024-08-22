use std::path::PathBuf;

use approx::assert_relative_eq;

use crate::evolutionary_models::{DNAModelType::*, FrequencyOptimisation, ProteinModelType::*};
use crate::likelihood::PhyloCostFunction;
use crate::optimisers::dna_model_optimiser::SubstModelOptimiser;
use crate::optimisers::ModelOptimiser;
use crate::phylo_info::PhyloInfoBuilder;
use crate::substitution_models::{
    DNALikelihoodCost, DNASubstModel, ProteinSubstModel, SubstLikelihoodCost, SubstitutionModel,
};

#[test]
fn check_likelihood_opt_k80() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let llik = DNALikelihoodCost::new(&model);
    // let cost =
    //     ProteinLikelihoodCost::new(&ProteinSubstModel::new(ProteinModelType::BLOSUM, &[]).unwrap());
    let unopt_logl = llik.cost(&info);
    let o = SubstModelOptimiser::new(&llik, &info, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_logl > unopt_logl);

    let model = DNASubstModel::new(K80, &[1.884815, 1.0]).unwrap();
    let expected_logl = DNALikelihoodCost::new(&model).cost(&info);

    assert_relative_eq!(o.final_logl, expected_logl, epsilon = 1e-6);
    assert_relative_eq!(o.final_logl, -4034.5008033, epsilon = 1e-6);
    assert_relative_eq!(o.model.params.rtc, 1.884815, epsilon = 1e-5);
    assert_relative_eq!(o.model.params.rta, 1.0, epsilon = 1e-5);
}

#[test]
fn frequencies_unchanged_opt_k80() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let o = SubstModelOptimiser::new(
        &DNALikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Empirical,
    )
    .run()
    .unwrap();
    assert!(o.model.params.pi.iter().all(|&x| x == 0.25));
}

#[test]
fn parameter_definition_after_optim_k80() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(K80, &[4.0, 1.0]).unwrap();
    let o = SubstModelOptimiser::new(
        &DNALikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    let params = &o.model.params;
    assert_eq!(params.rtc, params.rag);
    assert_eq!(params.rta, params.rtg);
    assert_eq!(params.rta, params.rca);
    assert_eq!(params.rta, params.rcg);
    assert!(params.pi.iter().all(|&x| x == 0.25));
}

#[test]
fn gtr_on_k80_data() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/K80/K80.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(
        GTR,
        &[0.25, 0.35, 0.3, 0.1, 0.88, 0.03, 0.00001, 0.07, 0.02, 1.0],
    )
    .unwrap();
    let o = SubstModelOptimiser::new(
        &DNALikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Empirical,
    )
    .run()
    .unwrap();
    let params = &o.model.params;
    assert_relative_eq!(params.rta, params.rtg, epsilon = 1e-1);
    assert_relative_eq!(params.rta, params.rca, epsilon = 1e-1);
    assert_relative_eq!(params.rta, params.rcg, epsilon = 1e-1);
    assert!(params.pi.iter().all(|&x| x - 0.25 < 1e-2));
}

#[test]
fn parameter_definition_after_optim_hky() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(HKY, &[0.26, 0.2, 0.4, 0.14, 4.0, 1.0]).unwrap();
    let llik = DNALikelihoodCost::new(&model);
    let start_logl = llik.cost(&info);
    let o = SubstModelOptimiser::new(&llik, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let params = &o.model.params;
    assert_eq!(params.rtc, params.rag);
    assert_eq!(params.rta, params.rtg);
    assert_eq!(params.rta, params.rca);
    assert_eq!(params.rta, params.rcg);
    assert!(params.pi.iter().all(|&x| x != 0.25));
    assert_eq!(o.initial_logl, start_logl);
    assert!(o.final_logl > start_logl);
}

#[test]
fn parameter_definition_after_optim_tn93() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model = DNASubstModel::new(TN93, &[0.26, 0.2, 0.4, 0.14, 4.0, 2.0, 1.0]).unwrap();
    let llik = DNALikelihoodCost::new(&model);
    let start_logl = llik.cost(&info);
    let o = SubstModelOptimiser::new(&llik, &info, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    let params = &o.model.params;
    assert_ne!(params.rtc, params.rag);
    assert_eq!(params.rta, params.rtg);
    assert_eq!(params.rta, params.rca);
    assert_eq!(params.rta, params.rcg);
    assert!(params.pi.iter().all(|&x| x != 0.25));
    assert!(o.final_logl > start_logl);
}

#[test]
fn check_parameter_optimisation_gtr() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = DNASubstModel::new(
        GTR,
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
    let phyml_logl = DNALikelihoodCost::new(&phyml_model).cost(&info);
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_model = DNASubstModel::new(
        GTR,
        &[
            0.25318, 0.32894, 0.31196, 0.10592, 0.88892, 0.03190, 0.00001, 0.07102, 0.02418, 1.0,
        ],
    )
    .unwrap(); // Original input to paml
    let paml_logl = DNALikelihoodCost::new(&paml_model).cost(&info);
    assert!(phyml_logl > paml_logl);

    let model = DNASubstModel::new(
        GTR,
        &[
            0.24720, 0.35320, 0.29540, 0.10420, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();
    let o = SubstModelOptimiser::new(
        &DNALikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert!(o.final_logl > phyml_logl);
    assert!(o.final_logl > paml_logl);

    let o2 = SubstModelOptimiser::new(
        &DNALikelihoodCost::new(&o.model),
        &info,
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert!(o2.final_logl >= o.final_logl);
    assert!(o2.iterations < 10);
}

#[test]
fn frequencies_fixed_opt_gtr() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sim/GTR/gtr.fasta"),
        PathBuf::from("./data/sim/tree.newick"),
    )
    .build()
    .unwrap();
    let model =
        DNASubstModel::new(GTR, &[0.25, 0.35, 0.3, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let o = SubstModelOptimiser::new(
        &DNALikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert!(o.model.params.pi.as_slice() == [0.25, 0.35, 0.3, 0.1]);
}

#[test]
fn frequencies_fixed_protein() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_protein1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let initial_llik = SubstLikelihoodCost::new(&model).cost(&info);
    let o = SubstModelOptimiser::new(
        &SubstLikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert_eq!(initial_llik, o.initial_logl);
    assert_eq!(initial_llik, o.final_logl);
    assert_eq!(model.freqs(), o.model.freqs());
}

#[test]
fn frequencies_empirical_protein() {
    let info = PhyloInfoBuilder::with_attrs(
        PathBuf::from("./data/sequences_protein1.fasta"),
        PathBuf::from("./data/tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = ProteinSubstModel::new(WAG, &[]).unwrap();
    let initial_llik = SubstLikelihoodCost::new(&model).cost(&info);
    let o = SubstModelOptimiser::new(
        &SubstLikelihoodCost::new(&model),
        &info,
        FrequencyOptimisation::Empirical,
    )
    .run()
    .unwrap();
    assert_ne!(initial_llik, o.initial_logl);
    assert_ne!(initial_llik, o.final_logl);
    assert_ne!(model.freqs(), o.model.freqs());
}
